# Generating Noisy Data using Deep Learning methods

A repository comprising of code for generation of noisy speech data from clean data using deep learning methods

The code makes use of the [official SinGAN implementaion](https://github.com/tamarott/SinGAN) to generate noisy spectrograms of audio data.
We make use of the Paint2Image module of SinGAN.

## How the files are structured:

- Use feat_extract.py to generate the spectrogram image that will be fed to singan for training.
- Use reading_spec_gen_by_sgan.ipynb to reconstuct the audio files from the output of singan.

**Note: Take a look at the "spec gen by singan" folder to get a better idea.**
**Note: Download the official SinGAN implementation from the link above**
