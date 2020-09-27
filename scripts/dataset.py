import logging
import os
import pickle
import time

import torch
from filelock import FileLock
from torch.utils.data import IterableDataset, DataLoader
import torch

from transformers.tokenization_utils import PreTrainedTokenizer
from transformers import Trainer
from tqdm import tqdm
from random import random, shuffle

logger = logging.getLogger(__name__)


def build_example(tokenizer, document, segment_length):
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


def create_dataset_cache(tokenizer: PreTrainedTokenizer, file_path, max_seq_length=512, min_seq_length=128, overwrite_cache=True):
    segment_length = (max_seq_length - 3) // 2

    directory, filename = os.path.split(file_path)
    cached_features_file = os.path.join(
        directory, 'cache', "cached_lm_{}_{}_{}".format(tokenizer.__class__.__name__, str(max_seq_length), filename,),
    )

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
                    while len(document) >= min_seq_length:
                        document, example = build_example(tokenizer, document, segment_length)
                        examples.append(example)
                    document = []
                document.extend(tokenizer.tokenize(line))
            while len(document) > min_seq_length:
                document, example = build_example(tokenizer, document, segment_length)
                examples.append(example)
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


def collate_batch(input, tokenizer, mlm_probability):
    batch = dict()
    for k, _ in input[0].items():
        batch[k] = torch.LongTensor([f[k] for f in input])
    
    if tokenizer.mask_token is None:
        raise ValueError(
            "This tokenizer does not have a mask token which is necessary for masked language modeling. Remove the --mlm flag if you want to use this tokenizer."
        )

    labels = batch['input_ids'].clone()
    inputs = batch['input_ids']
    # We sample a few tokens in each sequence for masked-LM training (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)
    probability_matrix = torch.full(labels.shape, mlm_probability)
    special_tokens_mask = [
        tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
    ]
    probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
    if tokenizer._pad_token is not None:
        padding_mask = labels.eq(tokenizer.pad_token_id)
        probability_matrix.masked_fill_(padding_mask, value=0.0)
    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels[~masked_indices] = -100  # We only compute loss on masked tokens

    # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
    inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

    # 10% of the time, we replace masked input tokens with random word
    indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
    random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long)
    inputs[indices_random] = random_words[indices_random]
    batch['input_ids'] = inputs
    batch['labels'] = labels
    # The rest of the time (10% of the time) we keep the masked input tokens unchanged
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
        if worker_info is None:
            self.files = parts
        else:
            worker_id = worker_info.id
            self.files = [f for f in parts if int(f[-2:]) % worker_info.num_workers == worker_id]
        self.files = [os.path.join(directory, f) for f in self.files]

    def __iter__(self):
        for f in self.files:
            with open(f, "rb") as handle:
                examples = pickle.load(handle)
                cnt = len(examples)
                #indices = shuffle(range(cnt))
                offset = 0
                while cnt - offset > self.batch_size:
                    yield collate_batch(examples[offset:offset + self.batch_size], self.tokenizer, self.mlm_probability)
                    offset += self.batch_size

    def __len__(self):
        return 20000 * len(self.files) // self.batch_size

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
            pin_memory=True
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
            pin_memory=True
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
            pin_memory=True
        )

        return data_loader
