
from params import *
checkpoint = get_last_checkpoint(output_dir)
print(checkpoint)
model = ConvbertForPreTraining.from_pretrained(checkpoint)
model.convbert.set_input_embeddings(model.convbert.get_input_embeddings())
model.save_pretrained(model_dir)


trainer = MyTrainer(model=model, args=training_args, train_dataset=train_dataset, prediction_loss_only=True, tb_writer=writer)

trainer.train(model_path=model_dir)
trainer.save_model(model_dir)
