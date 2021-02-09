import librosa
import numpy as np
import soundfile as sf
import torch
import sklearn.preprocessing as skp
import matplotlib.pyplot as plt
from scipy import signal


def calc_LSD(a, b):
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


def lag_finder(y1, y2, sr):
    n = len(y1)

    corr = signal.correlate(y2, y1, mode='same') / np.sqrt(signal.correlate(
        y1, y1, mode='same')[int(n/2)] * signal.correlate(y2, y2, mode='same')[int(n/2)])

    delay_arr = np.linspace(-0.5*n/sr, 0.5*n/sr, n)
    delay = delay_arr[np.argmax(corr)]
    print('y2 is ' + str(delay) + ' behind y1')

    """ plt.figure()
    plt.plot(delay_arr, corr)
    plt.title('Lag: ' + str(np.round(delay, 3)) + ' s')
    plt.xlabel('Lag')
    plt.ylabel('Correlation coeff')
    plt.show() """
    return delay


def normalize(sig1, sig2):
    """sig1 is samples of the ground_truth file
       sig2 is samples of the file to be normalized"""

    ##Apply Zero-Unit normalization
    sig1 = (sig1 - np.mean(sig1))/np.std(sig1)
    sig2 = (sig2 - np.mean(sig2))/np.std(sig2)

    ##Apply min_max normalization:
    sig1 = skp.minmax_scale(sig1)
    sig2 = skp.minmax_scale(sig2)

    return sig1, sig2


def norm_and_LSD(file1, file2):
    data1, sr1 = librosa.load(file1, sr=None)
    data2, sr2 = librosa.load(file2, sr=None)

    assert sr1 == sr2

    print(len(data1), len(data2))

    ##Padd with silence to make them equal
    zeros = np.zeros(np.abs((len(data2) - len(data1))), dtype=float)
    if(len(data1) < len(data2)):
        data1 = np.append(data1, zeros)
    elif(len(data2) < len(data1)):
        data2 = np.append(data2, zeros)

    #normalizing
    # Sig2 always the one to be normalized to match Sig1
    data1, data2 = normalize(sig1=data1, sig2=data2)

    """ ###Testing cross-correlation###########
    xcorr = np.correlate(data1, data2, "full")
    print(np.argmax(xcorr), type(xcorr) , np.max(xcorr))
    print("lag = ", np.argmax(xcorr) - xcorr.size//2) """

    if(len(zeros) != 0):
        delay = lag_finder(data1, data2, sr=sr1)
        data1 = np.roll(data1, int(delay*sr1))

        # To check if lag was fixed
        # ideally should be 0 or very close to zero
        lag_finder(data1, data2, sr=sr1)

    mag_spec1 = np.abs(librosa.stft(data1, n_fft=256, hop_length=128))**2
    mag_spec2 = np.abs(librosa.stft(data2, n_fft=256, hop_length=128))**2

    #print(mag_spec1.shape)
    #print(mag_spec1)

    mag_spec1 = librosa.power_to_db(mag_spec1)
    mag_spec2 = librosa.power_to_db(mag_spec2)

    a = torch.from_numpy(mag_spec1)
    b = torch.from_numpy(mag_spec2)

    print("LSD between %s, %s = %f" % (file1, file2, calc_LSD(a, b)))
    return
