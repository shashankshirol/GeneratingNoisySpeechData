import librosa
import numpy as np
import soundfile as sf
import torch
from timeit import default_timer as timer
from PIL import Image
import json

# @Author   : Shashank Shirol
# @Date     : 01/27/2021
# @File     : better_extract.py

# Using Librosa library to extract spec and phase information from an audio.
# Using it to reconstruct the audio. Also has code for calculating Log-spectral distance.
# Code for audio→spec→sinGAN→spec→audio

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

def extract(filename, n_fft = 256):
    """
        Extracts spectrogram from an input audio file
        Arguments:
            filename: path of the audio file
            n_fft: length of the windowed signal after padding with zeros.
    """
    data, sr = librosa.load(filename, sr=None)
    print(len(data))
    comp_spec = librosa.stft(data, n_fft=n_fft)

    mag_spec, phase = librosa.magphase(comp_spec)

    print(mag_spec.shape, type(mag_spec))
    print(phase.shape, type(phase))
    phase_in_angle = np.angle(phase)
    return mag_spec, phase_in_angle, sr

def reconstruct(mag_spec, phase):
    """
        Reconstructs frames from a spectrogram and phase information.
        Arguments:
            mag_spec: Magnitude component of a spectrogram
            phase:  Phase info. of a spectrogram
    """
    temp = mag_spec * np.exp(phase * 1j)
    data_out = librosa.istft(temp)
    return data_out


# to convert the spectrogram ( an 2d-array of real numbers) to a storable form (0-255)
def scale_minmax(X, min=0.0, max=1.0):
    X_std = (X - X.min()) / (X.max() - X.min())
    X_scaled = X_std * (max - min) + min
    return X_scaled, X.min(), X.max()


# to get the original spectrogram ( an 2d-array of real numbers) from an image form (0-255)
def unscale_minmax(X, X_min, X_max, min=0.0, max=1.0):
    X = X.astype(np.float)
    X -= min
    X /= (max - min)
    X *= (X_max - X_min)
    X += X_min
    return X


def to_rgb(im, rgb):  # converting the image into 3-channel for singan
    if(rgb == 1):
        return im
    w, h = im.shape
    ret = np.empty((w, h, rgb), dtype=np.uint8)
    ret[:, :, 0] = im
    for i in range(1, rgb):
        ret[:, :, i] = ret[:, :, 0]
    return ret

def save_spec_img(filename, mag_spec, phase, sr, clean = 0):
    name = filename.split('.')[0]

    #storing spec on db units(log based)
    mag_spec = librosa.power_to_db(mag_spec)

    X, X_min, X_max = scale_minmax(mag_spec, 0, 255)
    
    # Preserve orientation of the Spectrogram: lower freq at bottom, higher at the top
    X = np.flip(X, axis = 0)

    print(X_min, X_max)
    print(X.max(), X.min())

    if(clean):
        info_dict = {}
        info_dict['min'] = str(X_min)
        info_dict['max'] = str(X_max)

        phase_dict = {}
        phase_aslist = phase.tolist()
        phase_dict['phase'] = phase_aslist
        phase_dict['rate'] = sr

        with open(name + "_phase_data.json", "w") as op:
            json.dump(phase_dict, op)
        
        with open(name + "_info_data.json", "w") as op:
            json.dump(info_dict, op)
    else:
        name+="_noise"
    
    np_img = X.astype(np.uint8)
    rgb_im = to_rgb(np_img, 3)
    img = Image.fromarray(rgb_im)
    sav = name + ".png"
    img.save(sav)
    return

def reconstruct_from_image(filename, recon_filename):
    name = filename.split('.')[0]
    recon_name = recon_filename.split('.')[0]

    img = np.copy(np.asarray(Image.open(filename))).astype(np.float)  # loading image generated by singan
    print(img.shape)
    img = img[:, :, 0:3]  # dropping the 4th channel
    # converting the image into a 1-channel by taking mean across the 3-channels
    img = np.mean(img, axis=2)

    img = np.flip(img, axis=0)  #flipping the image to get the original representation

    with open(recon_name + "_info_data.json", "r") as ip:
        info_dict = json.load(ip)

    new_img = unscale_minmax(img, float(info_dict['min']), float(info_dict['max']), 0, 255)
    
    with open(recon_name + "_phase_data.json", "r") as ip:
        phase_dict = json.load(ip)
    phase = np.array(phase_dict['phase'])

    ## converting db to power for reconstruction
    spec = librosa.db_to_power(new_img)

    print(spec.shape)
    print(spec)

    startT = timer()
    data_out = reconstruct(spec, phase)
    endT = timer()
    print("Time taken to reconstruct = ", endT - startT)

    sav = name + '_reconstructed.wav'
    sf.write(sav, data_out, phase_dict['rate'])
    return

def norm_and_LSD(file1, file2):
    data1, sr1 = librosa.load(file1, sr=None)
    data2, sr2 = librosa.load(file2, sr=None)
    
    ####################################
    #normalising
    data1 = data1/data1.max()
    data2 = data2/data2.max()
    ####################################

    mag_spec1 = np.abs(librosa.stft(data1, n_fft=256))
    mag_spec2 = np.abs(librosa.stft(data2, n_fft=256))

    a = torch.from_numpy(librosa.power_to_db(mag_spec1**2))
    b = torch.from_numpy(librosa.power_to_db(mag_spec2**2))
    print("LSD = ", calc_LSD(a, b))
    return

def main():
    #file_name = "NEW_EX_conc\\RATs_noise_1_2_8k.wav"
    #mag_spec, phase, sr = extract(filename=file_name)
    #data_out = reconstruct(mag_spec, phase)
    #sf.write('recons.wav', data_out, sr)
    #power_spec_1 = librosa.power_to_db(S=mag_spec)

    #save_spec_img(file_name, mag_spec, phase, sr, clean=0)
    #reconstruct_from_image("RATs_noise_1_2_8k\start_scale=3.png", recon_filename="paint.png")

    #mag_spec, phase, sr = extract("female_noisy.wav")
    #power_spec_2 = librosa.power_to_db(S=mag_spec)


    #power_spec_1 = torch.from_numpy(power_spec_1)
    #power_spec_2 = torch.from_numpy(power_spec_2)

    #print(calc_LSD(power_spec_1, power_spec_2))
    norm_and_LSD("RATs_noise_1_2_8k\start_scale=3_reconstructed.wav", "paint.wav")
    pass

if __name__ == "__main__":
    main()
