  #!/usr/bin/env python
# -*- coding: utf-8 -*-

#Authors: Chng Eng Siong, Shashank Shirol
# extending Chenglin's code to extract features and generate signal
# we will use python_speech_feature directly
# as well as various matplotlib to show the signal of interest

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import  python_speech_features as psf
import  scipy.io.wavfile as wav
import  matplotlib.pyplot as plt
import sys
import argparse
import matplotlib
import  numpy as np
from scipy.signal import hamming
from timeit import default_timer as timer
from PIL import Image
import json

# This function uses scipy's wavfile routine to read.
# unfortunately scipy's function is primitive,
# we want to return a normalized value +- 1.0 in amplitude for processing

def audioread(filename):
    (rate,sig) = wav.read(filename)
    if sig.dtype == 'int16':
        nb_bits = 16
    elif sig.dtype == 'int32':
        nb_bits = 32
    max_nb_bit = float(2 ** (nb_bits - 1))
    sig = sig/(max_nb_bit+1.0)

    return rate, sig

def normhamming(fft_len):
    if fft_len == 512:
        frame_shift = 160
    elif fft_len == 256:
        frame_shift = 128
    else:
        print("Wrong fft_len, current only support 16k/8k sampling rate wav")
        exit(1)
    win = np.sqrt(hamming(fft_len, False))
    win = win/np.sqrt(np.sum(np.power(win[0:fft_len:frame_shift],2)))
    return win


def extract(filename, FFT_LEN, FRAME_SHIFT):
    # extract mag for mixture
    rate, sig = audioread(filename)
    frames = psf.sigproc.framesig(sig, FFT_LEN, FRAME_SHIFT, lambda x: normhamming(x))
#    frames = framesig(sig, FFT_LEN, FRAME_SHIFT, lambda x: normhamming(x))
#    phase, mag_spec = magspec(frames, FFT_LEN)
    complex_spec = np.fft.rfft(frames, FFT_LEN)
    phase        = np.angle(complex_spec)
    mag_spec     = np.absolute(complex_spec)

    return phase, mag_spec, rate, sig


def reconstruct(enhan_spec, phase, FFT_LEN, FRAME_SHIFT):

  # following is a sanity check, realising why ONLY strong spectral values
  # phase are important, and the rest we can set to zero. :)
  threshold_val = 0.001*enhan_spec.max()
  threshold     = threshold_val*np.ones(enhan_spec.shape)
  my_mask       = ((enhan_spec - threshold) > 0)
  my_phase      = phase * my_mask
  print('max enhan spec = ', enhan_spec.max(), 'and threshold =', threshold_val)
  nr,nc = my_mask.shape
  numElements = nr*nc
  print('number of spectra bin > threshold = ', my_mask.sum(),' as percentage = ',my_mask.sum()/numElements)

  """ fig, ax = plt.subplots(2, 1)
  ax[0].plot(my_mask)
  ax[1].plot(my_phase)
  plt.show() """

  spec_comp = enhan_spec * np.exp(my_phase * 1j)
  nb_bits = 16
  enhan_frames = np.fft.irfft(spec_comp)
  enhan_sig = psf.sigproc.deframesig(enhan_frames, 0, FFT_LEN, FRAME_SHIFT, lambda x: normhamming(x))
  enhan_sig = enhan_sig / np.max(np.abs(enhan_sig)) * 0.8
  # above is simply to get the amplitude to normalize to some reasonable value
  max_nb_bit = float(2 ** (nb_bits - 1))
  enhan_sig = enhan_sig * (max_nb_bit - 1.0)
  if nb_bits == 16:
      enhan_sig = enhan_sig.astype(np.int16)
  elif nb_bits == 32:
      enhan_sig = enhan_sig.astype(np.int32)

  return enhan_sig


######################################################################################

def scale_minmax(X, min = 0.0, max = 1.0):       ## to convert the spectrogram ( an 2d-array of floating point numbers) to a storable form (0-255)
    X_std = (X - X.min()) / (X.max() - X.min())
    X_scaled = X_std * (max - min) + min
    return X_scaled, X.min(), X.max()

def unscale_minmax(X, X_min, X_max, min=0.0, max=1.0):      ## to get the origiinal spectrogram ( an 2d-array of floating point numbers) from an image form (0-255)
    X = X.astype(np.float)
    X -= min
    X /= (max - min)
    X *= (X_max - X_min)
    X += X_min
    return X

def to_rgb(im, rgb):         ## converting the image into 3-channel for singan (if necessary)
    if(rgb == 1):
        return im
    w, h = im.shape
    ret = np.empty((w, h, rgb), dtype=np.uint8)
    ret[:, :, 0] = im
    for i in range(1, rgb):
        ret[:, :, i] = ret[:, :, 0]
    return ret

###################################################################################

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", help="Input Image file if reconstructing, else Input audio", type=str)
    parser.add_argument("--clean", "-c", help="Is Clean", type=int)
    parser.add_argument("--reconstruct", "-r", help="Reconstruct", type=int)
    parser.add_argument("--rgb", "-n", help="number of channels in image", type=int, default=1)
    parser.add_argument("--bytes", "-b", help="Bytes representation for image", type=int, default=1)
    args = parser.parse_args()

    if(args.input):
        orig_file =  args.input
        print(orig_file)
    else:
        sys.exit(1)
    
    if(args.clean):
        print(type(args.clean))
    
    #constants
    sample_rate  = 8000
    FFT_LEN      = 256
    FRAME_SHIFT  = 128

    if(args.reconstruct == 1): # Running the code in reconstruction mode
        img = np.copy(np.asarray(Image.open(orig_file))).astype(np.float)  # loading image generated by singan
        img = img[:, :, 0:3]  # dropping the 4th channel
        img = np.mean(img, axis=2)  # converting the image into a 1-channel by taking mean across the 3-channels

        with open("info_data.json", "r") as ip:
            info_dict = json.load(ip)
        new_img = unscale_minmax(img, info_dict['min'], info_dict['max'], info_dict['rep_min'], info_dict['rep_max'])

        with open("phase_data.json", "r") as ip:
            phase_dict = json.load(ip)
        phase = np.array(phase_dict['phase'])
        
        m_spec = new_img/20
        m_spec = 10**m_spec
        m_spec = np.transpose(m_spec)
        startT = timer()
        enhan = reconstruct(m_spec, phase, FFT_LEN, FRAME_SHIFT)
        endT = timer()
        print('Time Take reconstruct = ', endT-startT)


        #Saving the reconstructed audio data
        sav = orig_file.split('.')[0]+'_reconstructed.wav'
        wav.write(sav,phase_dict['rate'], enhan)
        #wav.write(sav, sample_rate, enhan)
        pass

    else: #Extracting spectrogram and storing as image
        # extract magnitude features
        phase, mag_spec, rate, sig = extract(orig_file, FFT_LEN, FRAME_SHIFT)
        nb_bits = 16
        print(mag_spec)
        print(mag_spec.shape)
        print('Signal datatype = ', sig.dtype)
        if sig.dtype == 'int16':
            nb_bits = 16
        elif sig.dtype == 'int32':
            nb_bits = 32
        
        mag_spec_np = np.copy(mag_spec.transpose())
        Y = (20 * np.log10(mag_spec_np)).clip(-90)  # The log-based
        rep_min = 0
        rep_max = 2**(args.bytes*8) - 1
        print('After Scale ------')
        X, X_min, X_max = scale_minmax(Y, rep_min, rep_max)
        print(X_max, X_min)

        info_dict = {}
        info_dict['min'] = X_min
        info_dict['max'] = X_max
        info_dict['rep_min'] = rep_min
        info_dict['rep_max'] = rep_max

        phase_dict = {}
        if(args.clean == True):
            phase_aslist = phase.tolist()
            phase_dict['phase'] = phase_aslist
            phase_dict['rate'] = rate
            with open("phase_data.json", "w") as op:
                json.dump(phase_dict, op)

        with open("info_data.json", "w") as op:
            json.dump(info_dict, op)

        np_img = X.astype(np.uint8)
        rgb_im = to_rgb(np_img, args.rgb)
        img = Image.fromarray(rgb_im)
        name = ""
        if(args.clean == True):
            name += "clean_"
        else:
            name += "noisy_"
        
        img.save(name+'out.png')
        pass


if __name__ == "__main__":
    main()
