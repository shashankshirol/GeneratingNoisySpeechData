from config import get_arguments
from SinGAN.manipulate import *
from SinGAN.training import *
import SinGAN.functions as functions


def adjust_scales2image(real_, opt):
    #opt.num_scales = int((math.log(math.pow(opt.min_size / (real_.shape[2]), 1), opt.scale_factor_init))) + 1
    opt.num_scales = math.ceil((math.log(math.pow(opt.min_size / (min(real_.shape[2], real_.shape[3])), 1), opt.scale_factor_init))) + 1
    scale2stop = math.ceil(math.log(min([opt.max_size, max([real_.shape[2], real_.shape[3]])]) / max([real_.shape[2], real_.shape[3]]), opt.scale_factor_init))
    opt.stop_scale = opt.num_scales - scale2stop
    # min(250/max([real_.shape[0],real_.shape[1]]),1)
    opt.scale1 = min(opt.max_size / max([real_.shape[2], real_.shape[3]]), 1)
    opt.scale1 = 1
    print("num scales = ", opt.num_scales)
    print("scale2stop = ", scale2stop)
    print("stop scale = ", opt.stop_scale)
    print("scale1 = ", opt.scale1)

    real = imresize(real_, opt.scale1, opt)
    print("new shape = ", real.shape)
    #opt.scale_factor = math.pow(opt.min_size / (real.shape[2]), 1 / (opt.stop_scale))
    opt.scale_factor = math.pow(
        opt.min_size/(min(real.shape[2], real.shape[3])), 1/(opt.stop_scale))
    scale2stop = math.ceil(math.log(min([opt.max_size, max(
        [real_.shape[2], real_.shape[3]])]) / max([real_.shape[2], real_.shape[3]]), opt.scale_factor_init))
    opt.stop_scale = opt.num_scales - scale2stop
    return real

def main():
    parser = get_arguments()
    parser.add_argument('--input_dir', help='input image dir', default='Input/Images')
    parser.add_argument('--input_name', help='input image name', required=True)
    parser.add_argument('--mode', help='task to be done', default='train')
    opt = parser.parse_args()
    opt = functions.post_config(opt)

    real = functions.read_image(opt)

    print(real.shape)
    adjust_scales2image(real, opt)
    pass

if __name__ == "__main__":
    main()
