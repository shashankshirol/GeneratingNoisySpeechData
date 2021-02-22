from config import get_arguments
from SinGAN.manipulate import *
from SinGAN.training import *
from SinGAN.imresize import imresize
from SinGAN.imresize import imresize_to_shape
from SinGAN.imresize import np2torch
import SinGAN.functions as functions
import soundfile as sf
import librosa

paint_size = []

if __name__ == '__main__':
    parser = get_arguments()
    parser.add_argument('--input_dir', help='input image dir', default='Input/Images')
    parser.add_argument('--input_wav_paint', help='paint wav file', required=True)
    parser.add_argument('--input_wav_train', help='train wav file', required=True)
    parser.add_argument('--input_name', help='training image name')
    parser.add_argument('--ref_dir', help='input reference dir', default='Input/Paint')
    parser.add_argument('--ref_name', help='reference image name')
    parser.add_argument('--paint_start_scale', help='paint injection scale', type=int, required=True)
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
    #dir2save = functions.generate_dir2save(opt)
    dir2save = '%s/Paint2image/%s/%s_out' % (opt.out, opt.input_wav_train[:-4],opt.input_wav_paint[:-4])
    if dir2save is None:
        print('task does not exist')
    #elif (os.path.exists(dir2save)):
    #    print("output already exist")
    else:
        try:
            os.makedirs(dir2save)
        except OSError:
            pass
        #real = functions.read_image(opt)

        #reading train_file
        real, _ = functions.read_wav_spec(opt.input_wav_train, opt)
        #real = functions.read_wav_melspec(opt.input_wav_train, opt)

        real = functions.adjust_scales2image(real, opt)
        Gs, Zs, reals, NoiseAmp = functions.load_trained_pyramid(opt)

        if (opt.paint_start_scale < 1) | (opt.paint_start_scale > (len(Gs)-1)):
            print("injection scale should be between 1 and %d" % (len(Gs)-1))
        else:

            #reading ref_image (paint):
            ref, ref_phase, ref_sr = functions.read_wav_spec(opt.input_wav_paint, opt, need_sr = True)

            if ref.shape[3] != real.shape[3]:

                ########################################################################################
                paint_size.append(ref.shape[2]) ##Saving paint_size for resizing the output
                paint_size.append(ref.shape[3])
                print("SIZE NOT SAME;")
                ########################################################################################

                ref = imresize_to_shape(ref, [real.shape[2], real.shape[3]], opt)
                ref = ref[:, :, :real.shape[2], :real.shape[3]]

            N = len(reals) - 1
            n = opt.paint_start_scale
            in_s = imresize(ref, pow(opt.scale_factor, (N - n + 1)), opt)
            in_s = in_s[:, :, :reals[n - 1].shape[2], :reals[n - 1].shape[3]]
            in_s = imresize(in_s, 1 / opt.scale_factor, opt)
            in_s = in_s[:, :, :reals[n].shape[2], :reals[n].shape[3]]
            
            print('\n')
            startT = timer()
            print('paint_scale=%d_start_timer=' % (opt.paint_start_scale), startT)
            out = SinGAN_generate(Gs[n:], Zs[n:], reals, NoiseAmp[n:], opt, in_s, n=n, num_samples=1)
            endT = timer()
            print('paint_scale=%d_end_timer=' % (opt.paint_start_scale), endT)
            print('paint_scale=%d_total_time = ' % (opt.paint_start_scale), endT-startT)
            print('\n')
            
            ####################################################################################################
            output_img = out.detach()
            inter = imresize_to_shape(output_img, paint_size, opt)
            spec_toSave = functions.convert_image_np(inter)
            #####################################################################################################

            #plt.imsave('%s/start_scale=%d.png' % (dir2save, opt.paint_start_scale), Image_toSave, vmin=0, vmax=1)
            print("Spec TO SAVE: ")
            print(spec_toSave.shape, type(spec_toSave))

            # Reconstructing audio:
            spec_toSave = spec_toSave[:, :, 0:3]
            spec_toSave = np.mean(spec_toSave, axis=2)
            data = functions.reconstruct_audio(spec_toSave, ref_phase)
            sf.write('%s/start_scale=%d.wav' % (dir2save, opt.paint_start_scale), data, ref_sr)