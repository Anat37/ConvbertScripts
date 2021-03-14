from transformers import Trainer, TrainingArguments
from transformers import AlbertTokenizer, ConvbertForPreTraining, ConvbertConfig
from transformers import AlbertForPreTraining, AlbertConfig
from dataset import SOPDataset, MyTrainer
import os


def get_last_checkpoint(dir_name):
    max_check = -1
    result = None
    for filename in os.listdir(dir_name):
        if 'checkpoint' in filename:
            step = int(filename.split('-')[1])
            if step > max_check:
                max_check = step
                result = filename
    return os.path.join(dir_name, result)


def init_convbert_model(config):
    model = ConvbertForPreTraining(config)
    ready_model = AlbertForPreTraining.from_pretrained('albert-base-v2')
    model.convbert.set_input_embeddings(ready_model.albert.get_input_embeddings())
    return model

def init_albert_model(config):
    model = AlbertForPreTraining(config)
    ready_model = AlbertForPreTraining.from_pretrained('albert-base-v2')
    model.albert.set_input_embeddings(ready_model.albert.get_input_embeddings())
    return model

def get_params(model_name, batch_size):
    model_dir = f'E:/ConvbertData/{model_name}/model_dir'
    output_dir = f'E:/ConvbertData/{model_name}/output'
    logs = f'E:/ConvbertData/{model_name}/logs'

    tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')
    tokenizer.save_pretrained(model_dir)
    train_dataset = SOPDataset(directory='E:/ConvbertData/text_data/cache', batch_size=batch_size, tokenizer=tokenizer, mlm_probability=0.15)

    training_args = TrainingArguments(
        output_dir=output_dir,          # output directory
        overwrite_output_dir=True,
        num_train_epochs=10,              # total # of training epochs
        per_device_train_batch_size=batch_size,  # batch size per device during training
        warmup_steps=500,                # number of warmup steps for learning rate scheduler
        weight_decay=0.01,               # strength of weight decay
        logging_dir=logs,            # directory for storing logs
        gradient_accumulation_steps=64,
        learning_rate=0.001506,
        dataloader_num_workers=3,
        logging_steps=100,
        save_total_limit=10,
        save_steps=200,
        adam_epsilon=1e-6
    )
    return training_args, train_dataset, model_dir, output_dir
