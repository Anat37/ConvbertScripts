from transformers import AlbertTokenizer, AlbertForPreTraining
import logging
from dataset import create_dataset_cache
import os

def main():
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)
    
    tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')
    data_path = 'D:/NIR/splitted/'
    for filename in os.listdir(data_path):
        create_dataset_cache(tokenizer, os.path.join(data_path, filename), overwrite_cache=False)


if __name__ == '__main__':
    main()