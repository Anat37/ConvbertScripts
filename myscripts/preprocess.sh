tensorboard --logdir=D:/ConvbertData/runs/
python ./transformers/examples/text-classification/run_glue.py --model_name_or_path 'E:/ConvbertData/convbert/model_dir' --task_name cola --do_train --do_eval --data_dir E:/ConvbertData/glue_data/CoLa --max_seq_length 512 --per_device_train_batch_size 16 --learning_rate 5e-6 --num_train_epochs 150.0 --output_dir E:/ConvbertData/glue_models/convbert/cola/ --overwrite_output_dir --per_device_eval_batch_size 128 --evaluate_during_training

python ./transformers/examples/text-classification/run_glue.py --model_name_or_path 'E:/ConvbertData/albert/model_dir' --task_name cola --do_train --do_eval --data_dir E:/ConvbertData/glue_data/CoLa --max_seq_length 512 --per_device_train_batch_size 16 --learning_rate 5e-6 --num_train_epochs 150.0 --output_dir E:/ConvbertData/glue_models/albert/cola/ --overwrite_output_dir --per_device_eval_batch_size 64 --evaluate_during_training



python ./transformers/examples/question-answering/run_squad.py --model_type convbert --model_name_or_path 'E:/ConvbertData/convbert/model_dir' --do_train --do_eval --do_lower_case --train_file E:/ConvbertData/squad_data/v2/train-v2.0.json --predict_file E:/ConvbertData/squad_data/v2/dev-v2.0.json --per_gpu_train_batch_size 16 --learning_rate 3e-5 --num_train_epochs 10.0 --max_seq_length 384 --doc_stride 128 --output_dir E:/ConvbertData/squad_models/conbert/v2/ --overwrite_output_dir --evaluate_during_training


python ./transformers/examples/question-answering/run_squad.py --model_type albert --model_name_or_path 'E:/ConvbertData/albert/model_dir' --do_train --do_eval --do_lower_case --train_file E:/ConvbertData/squad_data/v2/train-v2.0.json --predict_file E:/ConvbertData/squad_data/v2/dev-v2.0.json --per_gpu_train_batch_size 16 --learning_rate 3e-5 --num_train_epochs 3.0 --max_seq_length 384 --doc_stride 128 --output_dir E:/ConvbertData/squad_models/albert/v2/ --overwrite_output_dir