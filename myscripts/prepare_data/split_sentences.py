import os
import re

from blingfire import text_to_sentences, text_to_words

garbage = [r'\[\.\.\.\]', r'<.*?>', r'\(\*\)', r'\(\)']

def main():
    input_folder = 'E://NIR2/extracted'
    for dirname in os.listdir(input_folder):
        for filename in os.listdir(os.path.join(input_folder, dirname)):
            print('Processing ' + filename + ' from ' + dirname)
            with open(os.path.join(input_folder, dirname, filename), 'r', encoding='utf-8') as in_f:
                with open(os.path.join('splitted', dirname + filename), 'w', encoding='utf-8') as out_f:
                    lines = []
                    for line in in_f:
                        if '<doc' in line or line == '\n':
                            continue

                        if '</doc' in line:
                            if len(lines) > 3:
                                out_f.writelines(lines)
                                out_f.write('\n')
                            lines = []
                            continue
                        
                        # remove non-UTF
                        line = line.encode("ascii", "ignore").decode()
                        sentences = text_to_sentences(line).split('\n')
                        for s in sentences:
                            for ga in garbage:
                                s = re.sub(ga, '', s)
                            if len(s) <= 10:
                                continue
                            lines.append(s + '\n')

                        


if __name__ == "__main__":
    main()