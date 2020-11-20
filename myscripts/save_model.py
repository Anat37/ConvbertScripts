
from transformers import AutoModelForPreTraining
import os
import sys

model_dir = f'E:/ConvbertData/{sys.argv[1]}/model_dir'
output_dir = f'E:/ConvbertData/{sys.argv[1]}/output'


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

checkpoint = get_last_checkpoint(output_dir)
print(checkpoint)
model = AutoModelForPreTraining.from_pretrained(checkpoint)
model.save_pretrained(model_dir)
