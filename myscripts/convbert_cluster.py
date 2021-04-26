from transformers import Trainer, TrainingArguments
from transformers import AlbertTokenizer, ConvbertForPreTraining, ConvbertConfig
from transformers import AlbertForPreTraining, AlbertConfig, AlbertModel
from dataset import SOPDataset, MyTrainer
import os
import sys


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


def init_convbert_model(config, model):
    ready_model = AlbertModel.from_pretrained('albert-base-v2')
    model.convbert.set_input_embeddings(ready_model.get_input_embeddings())
    return model

def get_params(model_name, batch_size):
    model_dir = f'/home/akozlov/models/{model_name}/model_dir'
    output_dir = f'/home/akozlov/models/{model_name}/output'
    logs = f'/home/akozlov/models/{model_name}/logs'

    tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')
    tokenizer.save_pretrained(model_dir)
    train_dataset = SOPDataset(directory='/home/akozlov/data/cache', batch_size=batch_size, tokenizer=tokenizer, mlm_probability=0.15)

    training_args = TrainingArguments(
        output_dir=output_dir,          # output directory
        overwrite_output_dir=True,
        num_train_epochs=15,              # total # of training epochs
        per_device_train_batch_size=batch_size,  # batch size per device during training
        warmup_steps=500,                # number of warmup steps for learning rate scheduler
        weight_decay=0.01,               # strength of weight decay
        logging_dir=logs,            # directory for storing logs
        gradient_accumulation_steps=3*64,
        learning_rate=0.001506,
        dataloader_num_workers=3,
        logging_steps=100,
        save_total_limit=10,
        save_steps=200,
        adam_epsilon=1e-6
    )
    return training_args, train_dataset, model_dir, output_dir



def main():
    continuation = True
    training_args, train_dataset, model_dir, output_dir = get_params('convbert_12_127', 18)
    config = ConvbertConfig(hidden_size=768, num_attention_heads=12, intermediate_size=3072, attention_probs_dropout_prob=0, num_hidden_groups=12, num_hidden_layers=12, kernel_size=127)

    if continuation:
        checkpoint = get_last_checkpoint(output_dir)
        model = ConvbertForPreTraining.from_pretrained(checkpoint)
    else:
        config.save_pretrained(model_dir)
        model = ConvbertForPreTraining(config)

    model = init_convbert_model(config, model)
    model.save_pretrained(model_dir)

    trainer = MyTrainer(model=model, args=training_args, train_dataset=train_dataset, prediction_loss_only=True)
    print('kernel_size')
    print(model.convbert.encoder.kernel_size)
    if continuation:
        checkpoint = get_last_checkpoint(output_dir)
        trainer.train(model_path=model_dir)
    else:
        trainer.train(model_path=model_dir)
    trainer.save_model(model_dir)


if __name__ == '__main__':
    main()
