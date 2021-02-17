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
#from timeit import default_timer as timer


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

def calc_LSD_spectrum(a, b, n):
    """
        Computes LSD using the Power Spectrums (not Spectrograms)
        Arguments:
            a: (torch.Tensor), Power Spectrum for Signal 1
            b: (torch.Tensor), Power Spectrum for Signal 2
            n: length of either a/b (both are equal)
    """

    return torch.sqrt(torch.sum((a-b)**2)/n).numpy()


def AddNoiseFloor(data):
    frameSz = 64
    noiseFloor = (np.random.rand(frameSz, 1) - 0.5) * 1e-5
    numFrame = math.floor(len(data)/frameSz)
    st = 0
    et = frameSz

    for i in range(numFrame):
        if(np.sum(np.abs(data[st:et+1])) < 1e-5):
            data[st:et+1] += noiseFloor
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

    gain_d1d2 = np.mean(
        np.divide(sorted_data1[L1:L2+1], sorted_data2[L1:L2+1]))

    #Apply gain
    data2 = data2 * gain_d1d2

    return data1, data2


def normalize(sig1, sig2):
    """sig1 is the ground_truth file
       sig2 is the file to be normalized"""

    def get_mediainfo(sig):
        rate, data = wav.read(sig)
        bits_per_sample = np.NaN
        if(data.dtype == 'int16'):
            bits_per_sample = 16
        elif(data.dtype == 'int32'):
            bits_per_sample = 32

        return rate, bits_per_sample

    sample_rate1, bits_per_sample_sig1 = get_mediainfo(sig1)
    sample_rate2, bits_per_sample_sig2 = get_mediainfo(sig2)

    ## bps and sample rate must match
    assert bits_per_sample_sig1 == bits_per_sample_sig2
    assert sample_rate1 == sample_rate2

    def match_target_amplitude(sound, target):
        change = target - sound.dBFS
        return sound.apply_gain(change)

    sound1 = AudioSegment.from_wav(sig1)
    sound2 = AudioSegment.from_wav(sig2)

    ## Matching loudness
    sound2 = match_target_amplitude(sound2, sound1.dBFS)

    ## getting it back to librosa form
    samples1 = sound1.get_array_of_samples()
    data1 = np.array(samples1).astype(np.float32) / \
        (2**(bits_per_sample_sig1 - 1))

    samples2 = sound2.get_array_of_samples()
    data2 = np.array(samples2).astype(np.float32) / \
        (2**(bits_per_sample_sig2 - 1))

    return data1, data2, sample_rate1


def norm_and_LSD(file1, file2):
    nfft = 256
    overlapSz = 128
    frameSz = 256

    #normalizing
    #st = timer()
    # Sig2 always the one to be normalized to match Sig1
    data1, data2, sr = normalize(sig1=file1, sig2=file2)
    #en = timer()
    #print("Time for normalizing = ", en - st)

    """ ###Testing cross-correlation###########
    xcorr = np.correlate(data1, data2, "full")
    print(np.argmax(xcorr), type(xcorr) , np.max(xcorr))
    print("lag = ", np.argmax(xcorr) - xcorr.size//2) """

    data1, data2 = time_and_energy_align(data1, data2, sr=sr)
    assert len(data1) == len(data2)

    n = len(data1)

    s1_1 = abs(librosa.stft(data1, n_fft=nfft, hop_length=frameSz - overlapSz, win_length=frameSz, window='hamming'))**2 # Power Spectrogram
    s2_1 = abs(librosa.stft(data2, n_fft=nfft, hop_length=frameSz - overlapSz, win_length=frameSz, window='hamming'))**2 # Power Spectrogram

    _, s1_2 = signal.welch(data1, sr, window='hamming', nperseg=frameSz, noverlap=overlapSz, nfft=nfft, scaling='spectrum') # Power Spectrum
    _, s2_2 = signal.welch(data2, sr, window='hamming', nperseg=frameSz, noverlap=overlapSz, nfft=nfft, scaling='spectrum') # Power Spectrum

    # librosa.power_todb(S) basically returns 10*log10(S)
    s1_1 = librosa.power_to_db(s1_1)
    # librosa.power_todb(S) basically returns 10*log10(S)
    s2_1 = librosa.power_to_db(s2_1)

    # librosa.power_todb(S) basically returns 10*log10(S)
    s1_2 = librosa.power_to_db(s1_2)
    # librosa.power_todb(S) basically returns 10*log10(S)
    s2_2 = librosa.power_to_db(s2_2)

    a_1 = torch.from_numpy(s1_1)
    b_1 = torch.from_numpy(s2_1)

    a_2 = torch.from_numpy(s1_2)
    b_2 = torch.from_numpy(s2_2)

    print("LSD (Spectrogram) between %s, %s = %f" % (file1, file2, calc_LSD_spectrogram(a_1, b_1)))
    print("LSD (Spectrum) between %s, %s = %f" % (file1, file2, calc_LSD_spectrum(a_2, b_2, n)))
    return