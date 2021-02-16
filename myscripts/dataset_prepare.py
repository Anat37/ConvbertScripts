from transformers import AlbertTokenizer, AlbertForPreTraining
import logging
from dataset import create_dataset_cache
import os
import sys

def main():
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)

    base = 1
    part = 0
    argc = len(sys.argv)
    if argc > 1:
        base = int(sys.argv[1])
        if argc > 2:
            part = int(sys.argv[2])

    
    tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')
    data_path = 'E:/ConvbertData/text_data'
    for filename in os.listdir(data_path):
        fpath = os.path.join(data_path, filename)
        if not os.path.isfile(fpath):
            continue
        if int(filename[-2:]) % base != part:
            continue
        create_dataset_cache(tokenizer, fpath, overwrite_cache=False)


if __name__ == '__main__':
    main()