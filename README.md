# Generating Noisy Speech Data using Deep Learning methods

A repository comprising of code for generation of noisy speech data from clean data in the frequency domain using deep learning methods.

We explore two architectures -- one uses a style transfer method and the other uses an image-to-image translation model.

## Architectures:

### Style-Transfer Method:
The code makes use of the [official SinGAN implementaion](https://github.com/tamarott/SinGAN) to generate noisy spectrograms of audio data. We make use of the Paint2Image task of SinGAN.

### Image-to-Image Translation:
This repo houses a modified version of CUT: Contrastive unpaired Translation GAN which we use to learn a mapping from clean to noisy spectrograms. We have tuned the model enough for it to work on spectrograms and produce recontructable audio. The code is heavily derived from the official implementation available at [official CUT implementaion](https://github.com/taesungp/contrastive-unpaired-translation)

### Note
Refer to the directories pertaining to the two architectures to learn more and test them out for yourselves!

## Consolidated list of important links
- [SinGAN Paper](https://arxiv.org/pdf/1905.01164.pdf)
- [CUT Paper](https://arxiv.org/abs/2007.15651)
