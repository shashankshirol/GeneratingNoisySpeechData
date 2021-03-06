import librosa
import soundfile as sf
import numpy as np


def compute_mcd(data1, data2, sr):

    n_fft = 256
    win_length = 256
    hop_length = win_length // 4

    S1 = librosa.feature.melspectrogram(
        y=data1, sr=sr, n_fft=n_fft, hop_length=hop_length, window='hamming', n_mels=20, htk=True, norm=None)
    log_S1 = librosa.power_to_db(S1)/10  # computes log10(x)

    S2 = librosa.feature.melspectrogram(
        y=data2, sr=sr, n_fft=n_fft, hop_length=hop_length, window='hamming', n_mels=20, htk=True, norm=None)
    log_S2 = librosa.power_to_db(S2)/10  # computes log10(x)

    mfccs_S1 = librosa.feature.mfcc(S=log_S1, sr=sr)[1:, :]
    mfccs_S2 = librosa.feature.mfcc(S=log_S2, sr=sr)[1:, :]

    mfccs_diff = mfccs_S1 - mfccs_S2
    mfccs_diff_norm = np.linalg.norm(mfccs_diff, axis=1)
    mcd = np.mean(mfccs_diff_norm)
    no_frames = len(mfccs_diff_norm)

    return mcd, no_frames
