import logging
import os
import pickle
import time

import torch
from filelock import FileLock
from torch.utils.data import IterableDataset, DataLoader
import torch

from transformers.tokenization_utils import PreTrainedTokenizer
from transformers import Trainer, get_polynomial_decay_schedule_with_warmup
from tqdm import tqdm
from random import random, shuffle, randint
import numpy as np
import collections

from lamb import Lamb

logger = logging.getLogger(__name__)


def build_example(tokenizer, documents, total_len, max_seq_length):
    segment_length = (max_seq_length - 3) // 2
    if 2 * segment_length > len(document):
        segment_length = len(document) // 2
        part_2 = document[segment_length:]
    else:
        part_2 = document[segment_length:2*segment_length]
    part_1 = document[:segment_length]
    sop_label = 1
    document = document[len(part_1) + len(part_2):]
    if random() > 0.5:
        sop_label = 0
        part_1, part_2 = part_2, part_1
    example = tokenizer(part_1, part_2, is_split_into_words=True, truncation=True, padding='max_length')
    example['sentence_order_label'] = sop_label
    return document, example


def truncate_seq_pair(tokens_a, tokens_b, max_num_tokens):
  """Truncates a pair of sequences to a maximum sequence length."""
  while True:
    total_length = len(tokens_a) + len(tokens_b)
    if total_length <= max_num_tokens:
      break

    trunc_tokens = tokens_a if len(tokens_a) > len(tokens_b) else tokens_b
    assert len(trunc_tokens) >= 1

    # We want to sometimes truncate from the front and sometimes from the
    # back to add more randomness and avoid biases.
    if random() < 0.5:
      del trunc_tokens[0]
    else:
      trunc_tokens.pop()

MaskedLmInstance = collections.namedtuple("MaskedLmInstance", ["index", "label"])

def create_masked_lm_predictions(tokens, masked_lm_prob, max_predictions_per_seq, tokenizer):
  """Creates the predictions for the masked LM objective."""
  input_tokens = list(tokens)

  masked_lm_labels = []

  if masked_lm_prob == 0:
    return (input_tokens, masked_lm_labels)

  num_to_predict = min(max_predictions_per_seq,
                       max(1, int(round(len(tokens) * masked_lm_prob))))

  # Note(mingdachen):
  # By default, we set the probilities to favor shorter ngram sequences.
  ngrams = np.arange(1, 4, dtype=np.int64)
  pvals = 1. / np.arange(1, 4)
  pvals /= pvals.sum(keepdims=True)

  ngram_indexes = []
  for idx in range(len(tokens)):
    ngram_index = []
    for n in ngrams:
      ngram_index.append([[i] for i in range(idx, min(idx+n, len(tokens)))])
    ngram_indexes.append(ngram_index)
  
  shuffle(ngram_indexes)

  masked_lms = []
  covered_indexes = set()
  for cand_index_set in ngram_indexes:
    if len(masked_lms) >= num_to_predict:
      break
    if not cand_index_set:
      continue
    # Note(mingdachen):
    # Skip current piece if they are covered in lm masking or previous ngrams.
    for index_set in cand_index_set[0]:
      for index in index_set:
        if index in tokenizer.additional_special_tokens:
          continue
        if index in covered_indexes:
          continue

    n = np.random.choice(ngrams[:len(cand_index_set)],
                         p=pvals[:len(cand_index_set)] /
                         pvals[:len(cand_index_set)].sum(keepdims=True))
    index_set = sum(cand_index_set[n - 1], [])
    n -= 1
    # Note(mingdachen):
    # Repeatedly looking for a candidate that does not exceed the
    # maximum number of predictions by trying shorter ngrams.
    while len(masked_lms) + len(index_set) > num_to_predict:
      if n == 0:
        break
      index_set = sum(cand_index_set[n - 1], [])
      n -= 1
    # If adding a whole-word mask would exceed the maximum number of
    # predictions, then just skip this candidate.
    if len(masked_lms) + len(index_set) > num_to_predict:
      continue
    is_any_index_covered = False
    for index in index_set:
      if index in covered_indexes:
        is_any_index_covered = True
        break
    if is_any_index_covered:
      continue
    for index in index_set:
      covered_indexes.add(index)

      masked_token = None
      # 80% of the time, replace with [MASK]
      if random() < 0.8:
        masked_token = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)
      else:
        # 10% of the time, keep original
        if random() < 0.5:
          masked_token = tokens[index]
        # 10% of the time, replace with random word
        else:
          masked_token = randint(0, len(tokenizer) - 1)

      input_tokens[index] = masked_token

      masked_lms.append(MaskedLmInstance(index=index, label=tokens[index]))
  assert len(masked_lms) <= num_to_predict
  masked_lms = sorted(masked_lms, key=lambda x: x.index)
  labels = [-100] * len(input_tokens)
  for p in masked_lms:
    labels[p.index] = p.label

  return (input_tokens, labels)


def create_instances_from_document(document, tokenizer, masked_lm_prob, max_seq_length=512, max_predictions_per_seq=20, short_seq_prob=0.1):
  """Creates `TrainingInstance`s for a single document."""
  # Account for [CLS], [SEP], [SEP]
  max_num_tokens = max_seq_length - 3

  # We *usually* want to fill up the entire sequence since we are padding
  # to `max_seq_length` anyways, so short sequences are generally wasted
  # computation. However, we *sometimes*
  # (i.e., short_seq_prob == 0.1 == 10% of the time) want to use shorter
  # sequences to minimize the mismatch between pre-training and fine-tuning.
  # The `target_seq_length` is just a rough target however, whereas
  # `max_seq_length` is a hard limit.
  target_seq_length = max_num_tokens
  if random() < short_seq_prob:
    target_seq_length = randint(2, max_num_tokens)

  # We DON'T just concatenate all of the tokens from a document into a long
  # sequence and choose an arbitrary split point because this would make the
  # next sentence prediction task too easy. Instead, we split the input into
  # segments "A" and "B" based on the actual "sentences" provided by the user
  # input.
  instances = []
  current_chunk = []
  current_length = 0
  i = 0
  while i < len(document):
    segment = document[i]
    current_chunk.append(segment)
    current_length += len(segment)
    if i == len(document) - 1 or current_length >= target_seq_length:
      if current_chunk and len(current_chunk) > 1:
        # `a_end` is how many segments from `current_chunk` go into the `A`
        # (first) sentence.
        a_end = 1
        if len(current_chunk) >= 2:
          a_end = randint(1, len(current_chunk) - 1)

        tokens_a = []
        for j in range(a_end):
          tokens_a.extend(current_chunk[j])

        tokens_b = []
        for j in range(a_end, len(current_chunk)):
            tokens_b.extend(current_chunk[j])
        # Random next
        is_random_next = False
        if random() < 0.5:
          is_random_next = True
          # Note(mingdachen): in this case, we just swap tokens_a and tokens_b
          tokens_a, tokens_b = tokens_b, tokens_a
        # Actual next
        else:
          is_random_next = False
          
        truncate_seq_pair(tokens_a, tokens_b, max_num_tokens)

        assert len(tokens_a) >= 1
        assert len(tokens_b) >= 1
        
        example = tokenizer(tokens_a, tokens_b, is_split_into_words=True, truncation=True, padding='max_length')
        example['sentence_order_label'] = is_random_next

        #example["input_ids"], masked_lm_labels = create_masked_lm_predictions(example["input_ids"], masked_lm_prob, max_predictions_per_seq, tokenizer)
        #example["labels"] = masked_lm_labels
        #if random() > 0.99:
        #  print(example)
        instances.append(example)
      current_chunk = []
      current_length = 0
    i += 1

  return instances


def create_dataset_cache(tokenizer: PreTrainedTokenizer, file_path, masked_lm_prob=0.15, max_seq_length=512, min_seq_length=128, overwrite_cache=True):

    directory, filename = os.path.split(file_path)
    cached_features_file = os.path.join(
        directory, 'cache', "cached_lm_{}_{}_{}".format(tokenizer.__class__.__name__, str(max_seq_length), filename,),
    )
    if not os.path.exists(os.path.join(directory, 'cache')):
        os.makedirs(os.path.join(directory, 'cache'))

    if os.path.exists(cached_features_file) and not overwrite_cache:
        logger.info(f'Cache already exist {cached_features_file}')
    else:
        logger.info(f"Creating features from dataset file at {filename}")
        examples = []
        document = []
        start = time.time()
        with open(file_path, encoding="utf-8") as f:
            for line in tqdm(f):
                if line.strip() == '':
                    examples.extend(create_instances_from_document(document, tokenizer, masked_lm_prob, max_seq_length))
                    document = []
                else:
                    line = tokenizer.tokenize(line)
                    document.append(line)
        logger.info("Processed texts [took %.3f s]", time.time() - start)
        logger.info("Number of examples is %d", len(examples))

        start = time.time()
        with open(cached_features_file, "wb") as handle:
            pickle.dump(examples, handle, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info(
            "Saving features into cached file %s [took %.3f s]", cached_features_file, time.time() - start
        )


class SOPBatch():
    def __init__(self, data):
        self.data = data

    # custom memory pinning method on custom type
    def pin_memory(self):
        for k in self.data:
            self.data[k] = self.data[k].pin_memory()
        return self.data

    def to(self, device):
        for k in self.data:
            self.data[k] = self.data[k].to(device)
        return self.data


def old_collate_batch(input, tokenizer, mlm_probability):
    batch = dict()
    for k, _ in input[0].items():
        batch[k] = torch.LongTensor([f[k] for f in input])
    #print(len(input))
    
    if tokenizer.mask_token is None:
        raise ValueError(
            "This tokenizer does not have a mask token which is necessary for masked language modeling. Remove the --mlm flag if you want to use this tokenizer."
        )
    return SOPBatch(batch)


def collate_batch(input, tokenizer, mlm_probability):
    batch = dict()
    labeled_input = []
    for example in input:
        example["input_ids"], example["labels"] = create_masked_lm_predictions(example["input_ids"], mlm_probability, 20, tokenizer)
        labeled_input.append(example)
    for k, _ in labeled_input[0].items():
        batch[k] = torch.LongTensor([f[k] for f in labeled_input])
    #print(len(input))
    
    if tokenizer.mask_token is None:
        raise ValueError(
            "This tokenizer does not have a mask token which is necessary for masked language modeling. Remove the --mlm flag if you want to use this tokenizer."
        )
    return SOPBatch(batch)


class SOPDataset(IterableDataset):
    def __init__(self, directory: str, batch_size: int, tokenizer, mlm_probability):
        parts = os.listdir(directory)
        self.batch_size = batch_size
        self.tokenizer = tokenizer
        self.mlm_probability = mlm_probability
        if len(parts) == 0:
            logger.error("No cache found")
        worker_info = torch.utils.data.get_worker_info()
        self.all_files_cnt = len(parts)
        if worker_info is None:
            self.files = parts
        else:
            worker_id = worker_info.id
            self.files = [f for f in parts if (int(f[-2:]) % worker_info.num_workers) == worker_id]
        self.files = [os.path.join(directory, f) for f in self.files]
        shuffle(self.files)

    def __iter__(self):
        for f in self.files:
            with open(f, "rb") as handle:
                examples = pickle.load(handle)
                shuffle(examples)
                cnt = len(examples)
                offset = 0
                while cnt - offset > self.batch_size:
                    yield collate_batch(examples[offset:offset + self.batch_size], self.tokenizer, self.mlm_probability)
                    offset += self.batch_size

    def __len__(self):
        return 100000 * self.all_files_cnt // self.batch_size

def empty_collate(data):
    return data


class MyTrainer(Trainer):
    def get_train_dataloader(self):
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        data_loader = DataLoader(
            self.train_dataset,
            batch_size=None,
            collate_fn=empty_collate,
            pin_memory=True,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
        )

        return data_loader

    def get_eval_dataloader(self, eval_dataset = None):
        """
        Returns the evaluation :class:`~torch.utils.data.DataLoader`.
        Args:
            eval_dataset (:obj:`Dataset`, `optional`):
                If provided, will override `self.eval_dataset`.
        """
        if eval_dataset is None and self.eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")

        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset

        data_loader = DataLoader(
            eval_dataset,
            batch_size=None,
            collate_fn=empty_collate,
            pin_memory=True,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
        )

        return data_loader

    def get_test_dataloader(self, test_dataset):
        """
        Returns the test :class:`~torch.utils.data.DataLoader`.
        Args:
            test_dataset (obj:`Dataset`): The test dataset to use.
        """
        data_loader = DataLoader(
            test_dataset,
            batch_size=None,
            collate_fn=empty_collate,
            pin_memory=True,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
        )

        return data_loader

    def create_optimizer_and_scheduler(self, num_training_steps: int):
        """
        Setup the optimizer and the learning rate scheduler.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through :obj:`optimizers`, or subclass and override this method in a subclass.
        """
        if self.optimizer is None:
            no_decay = ["bias", "LayerNorm.weight"]
            optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                    "weight_decay": 0.0,
                },
            ]
            self.optimizer = Lamb(
                optimizer_grouped_parameters,
                lr=self.args.learning_rate,
                betas=(self.args.adam_beta1, self.args.adam_beta2),
                eps=self.args.adam_epsilon,
                weight_decay=self.args.weight_decay
            )
        if self.lr_scheduler is None:
            self.lr_scheduler = get_polynomial_decay_schedule_with_warmup(
                self.optimizer, num_warmup_steps=self.args.warmup_steps, num_training_steps=num_training_steps, lr_end=0.0
            )