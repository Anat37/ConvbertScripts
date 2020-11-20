from transformers import AlbertTokenizer, AlbertForPreTraining
import logging
from dataset import create_dataset_cache
import os

def main():
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)
    
    tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')
    data_path = 'E:/ConvbertData/text_data'
    for filename in os.listdir(data_path):
        fpath = os.path.join(data_path, filename)
        if not os.path.isfile(fpath):
            continue
        create_dataset_cache(tokenizer, fpath, overwrite_cache=False)


if __name__ == '__main__':
    main()