from params import *


def init_convbert_model(model):
    ready_model = AlbertForPreTraining.from_pretrained('albert-base-v2')
    model.convbert.set_input_embeddings(ready_model.albert.get_input_embeddings())
    return model

def main():
    training_args, train_dataset, model_dir, output_dir = get_params('convbert', 18)

    checkpoint = get_last_checkpoint(output_dir)
    
    model = ConvbertForPreTraining.from_pretrained(checkpoint)
    model = init_convbert_model(model)
    model.save_pretrained(model_dir)

    trainer = MyTrainer(model=model, args=training_args, train_dataset=train_dataset, prediction_loss_only=True)

    trainer.train(model_path=checkpoint)
    trainer.save_model(model_dir)


if __name__ == '__main__':
    main()
