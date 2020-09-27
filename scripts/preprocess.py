from transformers import BertForSequenceClassification, Trainer, TrainingArguments
from transformers import AlbertConfig, AlbertTokenizer, ConvbertForPreTraining
from transformers import TextDataset, DataCollatorForLanguageModeling
from dataset import SOPDataset, MyTrainer

model_dir = 'D:/ConvbertData/model_dir'
logs = 'D:/ConvbertData/logs'
runs = 'D:/ConvbertData/runs/convbert'

tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')
config = AlbertConfig(hidden_size=768, num_attention_heads=12, intermediate_size=2048)
model = ConvbertForPreTraining(config)
#model = ConvbertForPreTraining.from_pretrained(model_dir)
tokenizer.save_pretrained(model_dir)

train_dataset = SOPDataset(directory='D:/ConvbertData/text_data/cache', batch_size=8, tokenizer=tokenizer, mlm_probability=0.15)

training_args = TrainingArguments(
    output_dir=model_dir,          # output directory
    overwrite_output_dir=True,
    num_train_epochs=1,              # total # of training epochs
    per_device_train_batch_size=8,  # batch size per device during training
    warmup_steps=500,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    logging_dir=logs,            # directory for storing logs
)

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter(runs)

trainer = MyTrainer(model=model, args=training_args, train_dataset=train_dataset, prediction_loss_only=True, tb_writer=writer)

trainer.train(model_path=model_dir)
trainer.save_model()
