## How to use `clean_extract.py`:

- `python clean_extract.py -i <input_file> -c <1|0> -r <1|0> -b <int>`
- `-c is to denote if the input audio is the clean variant or not; decides whether phase info is stored or not.`
- `-r is to denote if running in reconstruction mode (from spec to audio) or not`
- `-b is number of byte representation for the spectrogram image; default is 1.`

*Note: when going from audio to spectrogram: input file is an audio;

when going from spectrogram to audio: input file is an image;*
