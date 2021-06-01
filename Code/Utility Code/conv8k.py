import os
import subprocess


def make_dataset(dir):
    files = []
    assert os.path.isdir(dir) or os.path.islink(
        dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir, followlinks=True)):
        for fname in fnames:
            path = os.path.join(root, fname)
            files.append(path)
    return files

if __name__ == "__main__":
    channels = ["B", "C", "E", "F", "G", "H"]
    path = "Ground_truths"
    for channel in channels[1:]:
        files = make_dataset(os.path.join(path, channel))
        for f in files:
            subprocess.call(['ffmpeg', '-hide_banner', '-loglevel', 'error',
                             '-i', f, '-ar', '8k', '-y', f[:-4] + '_8k.wav'])
            if(os.name == 'nt'):  # Windows
                os.system('del ' + f)
            else:  # Linux/MacOS/BSD
                os.system('rm ' + f)
            os.rename(f[:-4]+'_8k.wav', f)
