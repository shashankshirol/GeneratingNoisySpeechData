from __future__ import print_function
import SinGAN.functions
import SinGAN.models
import argparse
import os
import random
from SinGAN.imresize import imresize
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from skimage import io as img
import numpy as np
import soundfile as sf
from skimage import color
import librosa
import math
import imageio
import matplotlib.pyplot as plt
from SinGAN.training import *
from config import get_arguments

def SinGAN_generate(Gs,Zs,reals,NoiseAmp,opt,in_s=None,scale_v=1,scale_h=1,n=0,gen_start_scale=0,num_samples=50, phase=None, sr=None, extras = None):
    #if torch.is_tensor(in_s) == False:
    if in_s is None:
        in_s = torch.full(reals[0].shape, 0, device=opt.device)
    images_cur = []
    for G,Z_opt,noise_amp in zip(Gs,Zs,NoiseAmp):
        pad1 = ((opt.ker_size-1)*opt.num_layer)/2
        m = nn.ZeroPad2d(int(pad1))
        nzx = (Z_opt.shape[2]-pad1*2)*scale_v
        nzy = (Z_opt.shape[3]-pad1*2)*scale_h

        images_prev = images_cur
        images_cur = []

        for i in range(0,num_samples,1):
            if n == 0:
                z_curr = functions.generate_noise([1,nzx,nzy], device=opt.device)
                z_curr = z_curr.expand(1,3,z_curr.shape[2],z_curr.shape[3])
                z_curr = m(z_curr)
            else:
                z_curr = functions.generate_noise([opt.nc_z,nzx,nzy], device=opt.device)
                z_curr = m(z_curr)

            if images_prev == []:
                I_prev = m(in_s)
                #I_prev = m(I_prev)
                #I_prev = I_prev[:,:,0:z_curr.shape[2],0:z_curr.shape[3]]
                #I_prev = functions.upsampling(I_prev,z_curr.shape[2],z_curr.shape[3])
            else:
                I_prev = images_prev[i]
                I_prev = imresize(I_prev,1/opt.scale_factor, opt)
                I_prev = I_prev[:, :, 0:round(scale_v * reals[n].shape[2]), 0:round(scale_h * reals[n].shape[3])]
                I_prev = m(I_prev)
                I_prev = I_prev[:,:,0:z_curr.shape[2],0:z_curr.shape[3]]
                I_prev = functions.upsampling(I_prev,z_curr.shape[2],z_curr.shape[3])

            if n < gen_start_scale:
                z_curr = Z_opt

            z_in = noise_amp*(z_curr)+I_prev
            I_curr = G(z_in.detach(),I_prev)

            if n == len(reals)-1:
                if opt.mode == 'train':
                    dir2save = '%s/RandomSamples/%s/gen_start_scale=%d' % (opt.out, opt.input_name[:-4], gen_start_scale)
                else:
                    dir2save = functions.generate_dir2save(opt)
                if (opt.mode != "harmonization") & (opt.mode != "editing") & (opt.mode != "SR") & (opt.mode != "paint2image"):
                    if(phase is not None and sr is not None and extras is not None):
                        out_img = I_curr.detach()
                        spec_toSave = out_img[0, :, :, :, :]
                        spec_toSave = spec_toSave.permute((1, 2, 0))
                        spec_toSave = ((spec_toSave * 0.5) + 0.5) * 255
                        spec_toSave = spec_toSave.cpu().numpy().astype(np.uint8)

                        spec_toSave = spec_toSave[:, :, 0:3]
                        spec_toSave = np.mean(spec_toSave, axis=2)
                        spec_toSave = np.flip(spec_toSave, axis=0)
                        spec_toSave = functions.unscale_minmax(spec_toSave, float(extras[0]), float(extras[1]), 0, 255)
                        spec_toSave = librosa.db_to_power(spec_toSave)

                        data = functions.reconstruct_audio(spec_toSave, phase)
                        try:
                            os.makedirs(dir2save)
                        except OSError:
                            pass
                        sf.write('%s/randsample_%d.wav' % (dir2save, i), data, sr)
                    pass
            
            ## appending images to the list
            images_cur.append(I_curr)
        n+=1
    return I_curr.detach()
