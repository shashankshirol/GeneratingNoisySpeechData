import argparse
import os
import subprocess

CONST_FILES = 5  # because there are 5 files that will appear in EVERY model

def make_dataset(dir, max_dataset_size=float("inf")):
    files = []
    assert os.path.isdir(dir) or os.path.islink(
        dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir, followlinks=True)):
        for fname in fnames:
            path = os.path.join(root, fname)
            files.append(path)
    return files[:min(max_dataset_size, len(files))]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataroot', default='./dataset/content/', help = "Location of the test files")
    parser.add_argument('--name', type = str, help = "Name of the train file (used to store the model)", required = True)
    opt = parser.parse_args()


    #preparing test dataset
    files_loc = opt.dataroot
    files = make_dataset(files_loc)

    for f in files:
        print("Generating %s:" % (os.path.split(f)[-1]))
        subprocess.call(["python", "paint2image.py", "--input_wav_train", opt.name, "--input_wav_paint", f])

    return

if __name__ == "__main__":
    main()
