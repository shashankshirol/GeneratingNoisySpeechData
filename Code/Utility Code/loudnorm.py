import argparse
import pyloudnorm as pyln
import os
import soundfile as sf

FIXED_LOUDNESS = -23.0


def make_dataset(dir):
    files = []
    assert os.path.isdir(dir) or os.path.islink(
        dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir, followlinks=True)):
        for fname in fnames:
            path = os.path.join(root, fname)
            files.append(path)
    return files

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataroot", type=str, required=True, help="Location of the files")
    opt = parser.parse_args()
    
    files_loc = opt.dataroot
    files = make_dataset(files_loc)

    results = "RESULTS"
    try:
        os.mkdir(results)
    except OSError:
        print("Error in creating RESULTS directory")

    for f in files:
        file_name = os.path.split(f)[-1]
        data, sr = sf.read(f)

        meter = pyln.Meter(sr)
        loudness = meter.integrated_loudness(data)
        data_normalized = pyln.normalize.loudness(data, loudness, target_loudness = FIXED_LOUDNESS)
        sf.write(os.path.join(results, file_name), data_normalized, sr)
        print("Saved: %s" % (file_name))

    return

if __name__ == "__main__":
    main()
