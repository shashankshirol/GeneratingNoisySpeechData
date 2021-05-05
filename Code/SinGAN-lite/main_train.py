from config import get_arguments
from SinGAN.manipulate import *
from SinGAN.training import *
import SinGAN.functions as functions



if __name__ == '__main__':
    parser = get_arguments()
    parser.add_argument('--input_dir', help='input image dir', default='Input/Images')
    parser.add_argument('--input_name', help='input image name')
    parser.add_argument('--input_wav_train', help='train wav input', required=True)
    parser.add_argument('--mode', help='task to be done', default='train')
    opt = parser.parse_args()
    opt.input_name = opt.input_wav_train
    opt = functions.post_config(opt)
    
    Gs = []
    Zs = []
    reals = []
    NoiseAmp = []
    dir2save = functions.generate_dir2save(opt)

    if (os.path.exists(dir2save)):
        print('trained model already exist')
    else:
        try:
            os.makedirs(dir2save)
        except OSError:
            pass

        real, phase = functions.read_wav_spec(wav_file = opt.input_wav_train, opt = opt)
        
        print(real.shape, type(real))

        opt.max_size = max(real.shape[-2], real.shape[-1])
        
        functions.adjust_scales2image(real, opt)
        print(opt.stop_scale)
        train(opt, Gs, Zs, reals, NoiseAmp, real)
