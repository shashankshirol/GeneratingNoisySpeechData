import librosa
import numpy as np
import soundfile as sf
import torch
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav
from scipy import signal
from pydub import AudioSegment
import math
import random
import pyloudnorm as pyln
from timeit import default_timer as timer


def calc_LSD_spectrogram(a, b):
    """
        Computes LSD (Log - spectral distance)
        Arguments:
            a: vector (torch.Tensor), modified signal
            b: vector (torch.Tensor), reference signal (ground truth)
    """
    if(len(a) == len(b)):
        diff = torch.pow(a-b, 2)
    else:
        stop = min(len(a), len(b))
        diff = torch.pow(a[:stop] - b[:stop], 2)

    sum_freq = torch.sqrt(torch.sum(diff, dim=1)/diff.size(1))

    value = torch.sum(sum_freq, dim=0) / sum_freq.size(0)

    return value.numpy()


def AddNoiseFloor(data):
    frameSz = 128
    noiseFloor = (np.random.rand(frameSz) - 0.5) * 1e-5
    numFrame = math.floor(len(data)/frameSz)
    st = 0
    et = frameSz-1

    for i in range(numFrame):
        if(np.sum(np.abs(data[st:et+1])) < 1e-5):
            data[st:et+1] = data[st:et+1] + noiseFloor
        st = et + 1
        et += frameSz

    return data


def time_and_energy_align(data1, data2, sr):
    nfft = 256
    hop_length = 1  # hop_length = win_length or frameSz - overlapSz
    win_length = 256

    ##Adding small random noise to prevent -Inf problem with Spec
    data1 = AddNoiseFloor(data1)
    data2 = AddNoiseFloor(data2)

    ##Pad with silence to make them equal
    zeros = np.zeros(np.abs((len(data2) - len(data1))), dtype=float)
    padded = -1
    if(len(data1) < len(data2)):
        data1 = np.append(data1, zeros)
        padded = 1
    elif(len(data2) < len(data1)):
        data2 = np.append(data2, zeros)
        padded = 2
    
    
    # Time Alignment
    # Cross-Correlation and correction of lag using the spectrograms
    spec1 = abs(librosa.stft(data1, n_fft=nfft, hop_length=hop_length,
                             win_length=win_length, window='hamming'))
    spec2 = abs(librosa.stft(data2, n_fft=nfft, hop_length=hop_length,
                             win_length=win_length, window='hamming'))
    energy1 = np.mean(spec1, axis=0)
    energy2 = np.mean(spec2, axis=0)
    n = len(energy1)

    corr = signal.correlate(energy2, energy1, mode='same') / np.sqrt(signal.correlate(energy1,
                                                                                      energy1, mode='same')[int(n/2)] * signal.correlate(energy2, energy2, mode='same')[int(n/2)])
    delay_arr = np.linspace(-0.5*n/sr, 0.5*n/sr, n).round(decimals=6)

    #print(np.argmax(corr) - corr.size//2) no. of samples to move

    delay = delay_arr[np.argmax(corr)]
    print('y2 lags by ' + str(delay) + ' to y1')

    if(delay*sr < 0):
        to_roll = math.ceil(delay*sr)
    else:
        to_roll = math.floor(delay*sr)

    # correcting lag
    # if both signals were the same length, doesn't matter which one was rolled
    if(padded == 1 or padded == -1):
        data1 = np.roll(data1, to_roll)
    elif(padded == 2):
        data2 = np.roll(data2, -to_roll)

    #Plot Cross-correlation vs Lag; for debugging only;
    """ plt.figure()
    plt.plot(delay_arr, corr)
    plt.title('Lag: ' + str(np.round(delay, 3)) + ' s')
    plt.xlabel('Lag')
    plt.ylabel('Correlation coeff')
    plt.show() """

    # Energy Alignment

    data1 = data1 - np.mean(data1)
    data2 = data2 - np.mean(data2)

    sorted_data1 = -np.sort(-data1)
    sorted_data2 = -np.sort(-data2)

    L1 = math.floor(0.01*len(data1))
    L2 = math.floor(0.1*len(data1))

    gain_d1d2 = np.mean(np.divide(sorted_data1[L1:L2+1], sorted_data2[L1:L2+1]))

    #Apply gain
    data2 = data2 * gain_d1d2

    return data1, data2


def normalize(sig1, sig2):
    """sig1 is the ground_truth file
       sig2 is the file to be normalized"""

    data1, sr1 = sf.read(sig1)
    data2, sr2 = sf.read(sig2)

    assert sr1 == sr2

    meter1 = pyln.Meter(sr1)
    meter2 = pyln.Meter(sr2)

    loudness1 = meter1.integrated_loudness(data1)
    loudness2 = meter2.integrated_loudness(data2)

    print(loudness1, loudness2)

    data2_normalized = pyln.normalize.loudness(data2, loudness2, target_loudness=loudness1)

    return data1, data2_normalized, sr1


def norm_and_LSD(file1, file2):
    nfft = 256
    overlapSz = 128
    frameSz = 256
    eps = 1e-9
    # hop_length = frameSz - overlapSz; frameSz -> win_length

    #normalizing
    st = timer()
    # Sig2 always the one to be normalized to match Sig1
    data1, data2, sr = normalize(sig1=file1, sig2=file2)
    #data1, sr1 = librosa.load(file1, sr = 8000)
    #data2, sr2 = librosa.load(file2, sr = 8000)
    #sr = sr1 = sr2

    en = timer()
    print("Time for normalizing = ", en - st)
    
    """ ###Testing cross-correlation###########
    xcorr = np.correlate(data1, data2, "full")
    print(np.argmax(xcorr), type(xcorr) , np.max(xcorr))
    print("lag = ", np.argmax(xcorr) - xcorr.size//2) """

    data1, data2 = time_and_energy_align(data1, data2, sr=sr)
    assert len(data1) == len(data2)

    #sf.write(file2[:-4]+"_normed_"+file2[-4:], data2, sr)

    n = len(data1)

    s1 = (abs(librosa.stft(data1, n_fft=nfft, window='hamming'))**2)/n # Power Spectrogram
    s2 = (abs(librosa.stft(data2, n_fft=nfft, window='hamming'))**2)/n # Power Spectrogram

    # librosa.power_todb(S) basically returns 10*log10(S)
    s1 = librosa.power_to_db(s1 + eps)
    # librosa.power_todb(S) basically returns 10*log10(S)
    s2 = librosa.power_to_db(s2 + eps)

    a = torch.from_numpy(s1)
    b = torch.from_numpy(s2)

    print("LSD (Spectrogram) between %s, %s = %f" % (file1, file2, calc_LSD_spectrogram(a, b)))
    return

def main():

    f1 = "fe_03_1007-02235-A-033068-033931-A_8k.wav"
    f2 = "fe_03_1007-02235-A-033068-033931-src-comp_based-singChan_noNormTrain.wav"

    norm_and_LSD(f1, f2)

    return

if __name__ == "__main__":
    main()
