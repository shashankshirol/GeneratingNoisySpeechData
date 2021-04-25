# Contrastive Unpaired Translation (CUT)

This implementation of CUT borrows heavily from the [Official CUT](https://github.com/taesungp/contrastive-unpaired-translation) implementation. We introduce a few ease-of-use functionalities to serve our use-case. The details are briefly explained below.

## Implementational Changes:
- We include a few custom utility functions that help in dealing with audio files and process them internally.
- We include modules that take care of extraction of spectrogram in a form that is feedable to CUT, we also include modules that retain data required for reconstruction of output spectrogram.
- This version uses an updated data-loader that is tuned to work with Audio Spectrograms.
- We make use of a componentization module to split spectrograms of audio files into components that are treated as independent samples.
- To allow for a speedy processing, we make use of parallel processing capabilities.

## Goal:
- To learn a mapping from clean to noisy audio ONLY from their respective spectrograms. Once this mapping is learned, we can use the model to produce more noisy data given unseen clean data.
