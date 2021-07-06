import os
import sys
import subprocess

def main(i):
    seed = 34 + i * 10
    mode = sys.argv[3]
    if not os.path.exists('data{}'.format(i)):
        os.makedirs('data{}'.format(i))
    if mode == 'gut':
        for filename in os.listdir('Gutenberg/splitted2'):
            print('Processing ' + filename)
            if os.path.exists("./data{}/{}.tfrecord".format(i, filename)):
                continue
            subprocess.run([
                "python", "-m",
                "albert.create_pretraining_data",
                "--input_file=./Gutenberg/splitted2/{}".format(filename),
                "--output_file=./data{}/{}.tfrecord".format(i, filename),
                "--vocab_file=30k-clean.vocab",
                "--do_lower_case=True",
                "--max_seq_length=512",
                "--random_seed={}".format(seed),
                "--spm_model_file=30k-clean.model",
                "--dupe_factor=1"])
        return
    for filename in os.listdir('splitted'):
        num = int(filename[-2:])
        if num % 4 != int(mode):
            continue
        if os.path.exists("./data{}/{}.tfrecord".format(i, filename)):
            continue
        print('Processing ' + filename)
        subprocess.run([
            "python", "-m",
            "albert.create_pretraining_data",
            "--input_file=./splitted/{}".format(filename),
            "--output_file=./data{}/{}.tfrecord".format(i, filename),
            "--vocab_file=30k-clean.vocab",
            "--do_lower_case=True",
            "--max_seq_length=512",
            "--random_seed={}".format(seed),
            "--spm_model_file=30k-clean.model",
            "--dupe_factor=1"])
    


if __name__ == "__main__":
    for i in range(int(sys.argv[1]), int(sys.argv[2])):
        main(i)