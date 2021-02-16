from params import *


def main():
    training_args, train_dataset, model_dir, _ = get_params('convbert', 16)
    config = ConvbertConfig(hidden_size=768, num_attention_heads=12, intermediate_size=3072, attention_probs_dropout_prob=0, num_hidden_groups=4, num_hidden_layers=12, kernel_size=255)
    config.save_pretrained(model_dir)
    model = init_convbert_model(config)
    model.save_pretrained(model_dir)

    trainer = MyTrainer(model=model, args=training_args, train_dataset=train_dataset, prediction_loss_only=True)

    trainer.train(model_path=model_dir)
    trainer.save_model(model_dir)


if __name__ == '__main__':
    main()
