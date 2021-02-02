import scipy.io.wavfile as wav
import numpy as np
def audioread(filename):
    (rate, sig) = wav.read(filename)
    if sig.dtype == 'int16':
        nb_bits = 16
    elif sig.dtype == 'int32':
        nb_bits = 32
    max_nb_bit = float(2 ** (nb_bits - 1))
    sig = sig/(max_nb_bit+1.0)

    return rate, sig

def clip(sig, val):
    max_val = val*max(abs(sig))
    for i in range(len(sig)):
        if(abs(sig[i]) > max_val):
            if(sig[i] < 0):
                sig[i] = -max_val
            else:
                sig[i] = max_val

    return sig


def main():
    input_file = "output\start_scale=3_reconstructed.wav"
    rate, sig = audioread(input_file)
    val = 0.1
    rate, signal = wav.read(input_file)
    sig = np.copy(signal)
    sig = clip(sig, val)
    sav = input_file.split('.')[0]+'_'+str(val)+'_clipped.wav'
    wav.write(sav, rate, sig)
    pass

if __name__ == "__main__":
    main()
