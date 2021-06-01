import os
import soundfile as sf
import shutil


def main():
    ROOT = "RATS_allChann"

    #Required files
    TRAIN_ROOT = "RATS_training_corpus"
    clean = os.path.join(TRAIN_ROOT, "trainA")
    clean_files = os.listdir(clean)
    #Required channels
    channels = ["B", "C", "E", "F", "G", "H"]

    for channel in channels:
        os.mkdir(channel)
        for clean_file in clean_files:
            path = os.path.join(ROOT, "noisy_ch_"+channel, clean_file.replace("src", channel))
            shutil.copyfile(path, os.path.join(channel, os.path.split(path)[1]))

    pass

if __name__ == "__main__":
    main()
