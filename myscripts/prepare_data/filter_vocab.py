import os
import re

def has_cyrillic(text):
    return bool(re.search('[а-яА-Я]', text))

def has_digits(text):
    return bool(re.search(r'\d', text))

def main():
    vocab = set()
    with open('vocab.txt', 'r', encoding='utf-8') as f:
        for line in f:
            word = line.strip()
            if has_digits(word) or has_cyrillic(word):
                continue
            vocab.add(word)

    with open('vocab.txt', 'w', encoding='utf-8') as f:
        for word in vocab:
            f.write(word + '\n')

if __name__ == "__main__":
    main()