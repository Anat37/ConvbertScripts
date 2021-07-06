import os

from blingfire import text_to_sentences, text_to_words

def main():
    limit = 1 * 1000 * 1000
    i = 0
    cnt = 0
    folder = 'splitted'
    base_name = 'wiki0000'
    out_f = open(os.path.join(folder, 'splitted', base_name + str(i)), 'w', encoding='utf-8')
    for filename in os.listdir(folder):
        print('Processing ' + filename) 
        if int(filename[-2:]) < 29:
            continue
        if filename[:2] == 'AA':
            continue
        with open(os.path.join(folder, filename), 'r', encoding='utf-8') as in_f:
            lines = []
            for line in in_f:
                if line != '\n':
                    lines.append(line)
                    cnt += len(line)
                else:
                    out_f.writelines(lines)
                    out_f.write('\n')
                    lines = []
                    if cnt >= limit:
                        cnt = 0
                        out_f.close()
                        i += 1
                        out_f = open(os.path.join(folder, 'splitted', base_name + str(i)), 'w', encoding='utf-8')
    if lines:
        out_f.writelines(lines)
        out_f.write('\n')
    out_f.close()

if __name__ == "__main__":
    main()