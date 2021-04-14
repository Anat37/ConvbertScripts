from params import *


def init_convbert_model(model):
    ready_model = AlbertForPreTraining.from_pretrained('albert-base-v2')
    model.convbert.set_input_embeddings(ready_model.albert.get_input_embeddings())
    return model

def main():
    training_args, train_dataset, model_dir, output_dir = get_params('convbert_2', 18)

    checkpoint = get_last_checkpoint(output_dir)
    #checkpoint = get_last_checkpoint('E:/ConvbertData/convbert_test/output')
    training_args.num_train_epochs = 4
    #training_args.learning_rate = 0.000506

    model = ConvbertForPreTraining.from_pretrained(checkpoint)
    model = init_convbert_model(model)
    model.save_pretrained(model_dir)

    trainer = MyTrainer(model=model, args=training_args, train_dataset=train_dataset, prediction_loss_only=True)

    trainer.train(model_path=checkpoint)
    trainer.save_model(model_dir)


if __name__ == '__main__':
    main()
