import torch
import example_module
#import dynamicconv_cuda

from transformers import BertForSequenceClassification, Trainer, TrainingArguments
from transformers import AlbertConfig, AlbertTokenizer, AlbertForPreTraining, ConvbertForPreTraining, ConvbertConfig
from transformers import TextDataset, DataCollatorForLanguageModeling
from dataset import SOPDataset, MyTrainer
import os

model_dir = 'E:/ConvbertData/convbert/output'
#model_dir = 'E:/ConvbertData/albert_model_dir'
logs = 'E:/ConvbertData/logs'
runs = 'E:/ConvbertData/runs/convbert'
#runs = 'E:/ConvbertData/runs/albert'


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

tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')
#config = ConvbertConfig(hidden_size=768, num_attention_heads=12, intermediate_size=3072, attention_probs_dropout_prob=0, num_hidden_groups=4, num_hidden_layers=4)
#config.save_pretrained(model_dir)
#model = ConvbertForPreTraining(config)
#model = AlbertForPreTraining(config)
#model = AlbertForPreTraining.from_pretrained('albert-base-v2')
model = ConvbertForPreTraining.from_pretrained(get_last_checkpoint(model_dir))
#model.save_pretrained(model_dir)
#model = AlbertForPreTraining.from_pretrained(model_dir)
#tokenizer.save_pretrained(model_dir)

train_dataset = SOPDataset(directory='E:/ConvbertData/text_data/cache', batch_size=1, tokenizer=tokenizer, mlm_probability=0.1)

training_args = TrainingArguments(
    output_dir=model_dir,          # output directory
    overwrite_output_dir=True,
    num_train_epochs=5,              # total # of training epochs
    per_device_train_batch_size=28,  # batch size per device during training
    warmup_steps=500,                # number of warmup steps for learning rate scheduler
    weight_decay=0.001,               # strength of weight decay
    logging_dir=logs,            # directory for storing logs
    gradient_accumulation_steps=4,
    learning_rate=0.001,
    dataloader_num_workers=2
)

#from torch.utils.tensorboard import SummaryWriter
#writer = SummaryWriter(runs)

#trainer = MyTrainer(model=model, args=training_args, train_dataset=train_dataset, prediction_loss_only=True, tb_writer=writer)

#trainer.train(model_path=model_dir)
#trainer.save_model()
