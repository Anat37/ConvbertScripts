from transformers import BertForSequenceClassification, Trainer, TrainingArguments
from transformers import AlbertConfig, AlbertTokenizer, ConvbertForPreTraining
from transformers import TextDataset, DataCollatorForLanguageModeling
from dataset import SOPDataset, MyTrainer

tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')
config = AlbertConfig(hidden_size=768, num_attention_heads=12, intermediate_size=2048)
#model = ConvbertForPreTraining()
model = ConvbertForPreTraining.from_pretrained('./results')
tokenizer.save_pretrained('./results')

train_dataset = SOPDataset(directory='D:/NIR/splitted/cache', batch_size=2, tokenizer=tokenizer, mlm_probability=0.15)

training_args = TrainingArguments(
    output_dir='./results',          # output directory
    overwrite_output_dir=True,
    num_train_epochs=1,              # total # of training epochs
    per_device_train_batch_size=8,  # batch size per device during training
    warmup_steps=500,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    logging_dir='./logs',            # directory for storing logs
)

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('runs/convbert')

trainer = MyTrainer(model=model, args=training_args, train_dataset=train_dataset, prediction_loss_only=True, tb_writer=writer)

print('train')
#trainer.train(model_path='./model_path')
trainer.save_model()
