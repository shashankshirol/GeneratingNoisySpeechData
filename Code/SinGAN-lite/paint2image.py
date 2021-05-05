from config import get_arguments
from SinGAN.manipulate import *
from SinGAN.training import *
from SinGAN.imresize import imresize
from SinGAN.imresize import imresize_to_shape
from SinGAN.imresize import np2torch
import SinGAN.functions as functions
import soundfile as sf
import librosa
import numpy as np
import torch
import os

paint_size = []

if __name__ == '__main__':
    parser = get_arguments()
    parser.add_argument('--input_dir', help='input image dir', default='Input/Images')
    parser.add_argument('--input_wav_paint', help='paint wav file', required=True)
    parser.add_argument('--input_wav_train', help='train wav file', required=True)
    parser.add_argument('--input_name', help='training image name')
    parser.add_argument('--ref_dir', help='input reference dir', default='Input/Paint')
    parser.add_argument('--ref_name', help='reference image name')
    parser.add_argument('--paint_start_scale', help='paint injection scale', type=int)
    parser.add_argument('--quantization_flag', help='specify if to perform color quantization training', type=bool, default=False)
    parser.add_argument('--mode', help='task to be done', default='paint2image')
    opt = parser.parse_args()
    opt.input_name = opt.input_wav_train
    opt.ref_name = opt.input_wav_paint
    opt = functions.post_config(opt)

    Gs = []
    Zs = []
    reals = []
    NoiseAmp = []
    
    dir2save = '%s/Paint2image/%s/%s_out' % (opt.out, opt.input_wav_train[:-4],os.path.split(opt.input_wav_paint)[-1][:-4])
    if dir2save is None:
        print('task does not exist')
    else:
        try:
            os.mkdir(dir2save)
        except OSError:
            pass

        #reading train_file
        real, _ = functions.read_wav_spec(wav_file = opt.input_wav_train, opt = opt)
        opt.max_size = max(real.shape[-2], real.shape[-1])

        real = functions.adjust_scales2image(real, opt)
        Gs, Zs, reals, NoiseAmp = functions.load_trained_pyramid(opt)

        ref, ref_phase, ref_sr, ref_items = functions.read_wav_spec(wav_file=opt.input_wav_paint, opt = opt, need_sr = True, send_extra=True)

        ## for reconstruction
        ref_min = ref_items[0]
        ref_max = ref_items[1]

        if ref.shape[3] != real.shape[3]:
            ########################################################################################
            paint_size.append(ref.shape[2]) ##Saving paint_size for resizing the output
            paint_size.append(ref.shape[3])
            ########################################################################################

            ref = imresize_to_shape(ref, [real.shape[2], real.shape[3]], opt)
            ref = ref[:, :, :real.shape[2], :real.shape[3]]
        
        for scale in range(len(Gs) - 1): #generating output for all scales
            N = len(reals) - 1
            n = scale + 1
            in_s = imresize(ref, pow(opt.scale_factor, (N - n + 1)), opt)
            in_s = in_s[:, :, :reals[n - 1].shape[2], :reals[n - 1].shape[3]]
            in_s = imresize(in_s, 1 / opt.scale_factor, opt)
            in_s = in_s[:, :, :reals[n].shape[2], :reals[n].shape[3]]
            
            out = SinGAN_generate(Gs[n:], Zs[n:], reals, NoiseAmp[n:], opt, in_s, n=n, num_samples=1)
            
            
            ####################################################################################################
            output_img = out.detach()
            inter = imresize_to_shape(output_img, paint_size, opt)
            spec_toSave = inter[0, :, :, :]
            spec_toSave = spec_toSave.permute((1,2,0))
            spec_toSave = ((spec_toSave * 0.5) + 0.5) * 255
            spec_toSave = spec_toSave.cpu().numpy().astype(np.uint8)
            #####################################################################################################


            #### Reconstructing audio:

            spec_toSave = spec_toSave[:, :, 0:3]
            spec_toSave = np.mean(spec_toSave, axis=2)
            spec_toSave = np.flip(spec_toSave, axis=0)

            spec_toSave = functions.unscale_minmax(spec_toSave, float(ref_min), float(ref_max), 0, 255)
            spec_toSave = librosa.db_to_power(spec_toSave)

            data = functions.reconstruct_audio(spec_toSave, ref_phase)
            sf.write('%s/%s_start_scale=%d.wav' % (dir2save, os.path.split(opt.input_wav_paint)[-1][:-4], scale + 1), data, ref_sr)
