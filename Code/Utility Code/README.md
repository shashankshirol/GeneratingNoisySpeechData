# Loudness Normalization and LSD calculation

This directory houses some utility python scripts for loudness normalization and LSD calculations. 

## Loundness Normalization:
	Usage:
- `python loudnorm.py --dataroot <location of the audio files>`
*The resultant files are stored in the current working directory under a new folder, "RESULTS".*

### Note
The script normalizes the audio files to a fixed LUFS value of -23.0 dB (this can be changed from inside the script)

## LSD calculation
	Usage:
- `python LSD_cal.py`
*The LSD score is calculated between the two files mentioned inside the script.*

### Note
To calculate the LSD score for your files, change the file names accordingly inside the script.
