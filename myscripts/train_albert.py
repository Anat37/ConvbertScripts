from transformers import Trainer, TrainingArguments
from transformers import AlbertTokenizer, AlbertForPreTraining, AlbertConfig
from dataset import SOPDataset, MyTrainer
import os

model_dir = 'E:/ConvbertData/albert/model_dir'
output_dir = 'E:/ConvbertData/albert/output'
logs = 'E:/ConvbertData/albert/logs'
runs = 'E:/ConvbertData/albert/runs'

def get_last_checkpoint(dir_name):
    max_check = -1
    result = 'none'
    for filename in os.listdir(dir_name):
        if 'checkpoint' in filename:
            step = int(filename.split('-')[1])
            if step > max_check:
                max_check = step
                result = filename
    return os.path.join(dir_name, result)

checkpoint = get_last_checkpoint(output_dir)
print(checkpoint)

BATCH_SIZE = 16

tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')
tokenizer.save_pretrained(model_dir)
train_dataset = SOPDataset(directory='E:/ConvbertData/text_data/cache', batch_size=BATCH_SIZE, tokenizer=tokenizer, mlm_probability=0.04)

training_args = TrainingArguments(
    output_dir=output_dir,          # output directory
    overwrite_output_dir=True,
    num_train_epochs=10,              # total # of training epochs
    per_device_train_batch_size=BATCH_SIZE,  # batch size per device during training
    warmup_steps=2500,                # number of warmup steps for learning rate scheduler
    weight_decay=0.00001,               # strength of weight decay
    logging_dir=logs,            # directory for storing logs
    gradient_accumulation_steps=4*64,
    learning_rate=0.00176,
    dataloader_num_workers=2
)

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter(runs)



config = AlbertConfig(hidden_size=768, num_attention_heads=12, intermediate_size=3072, attention_probs_dropout_prob=0, num_hidden_groups=1, num_hidden_layers=12)
config.save_pretrained(model_dir)
model = AlbertForPreTraining(config)
#model = AlbertForPreTraining.from_pretrained(checkpoint)
model.save_pretrained(model_dir)



trainer = MyTrainer(model=model, args=training_args, train_dataset=train_dataset, prediction_loss_only=True, tb_writer=writer)

trainer.train(model_path=model_dir)
trainer.save_model(model_dir)

