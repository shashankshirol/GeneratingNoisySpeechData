# SinGAN

[Project](https://tamarott.github.io/SinGAN.htm) | [Arxiv](https://arxiv.org/pdf/1905.01164.pdf) | [CVF](http://openaccess.thecvf.com/content_ICCV_2019/papers/Shaham_SinGAN_Learning_a_Generative_Model_From_a_Single_Natural_Image_ICCV_2019_paper.pdf) 
### Non - Official pytorch implementation of the paper: "SinGAN: Learning a Generative Model from a Single Natural Image", ICCV 2019 Best paper award (Marr prize)


## Random samples from a *single* image
With SinGAN, you can train a generative model from a single natural image, and then generate random samples form the given image, for example:



## SinGAN's applications
SinGAN can be also use to a line of image manipulation task, for example:
This is done by injecting an image to the already trained model. See section 4 in the [paper](https://arxiv.org/pdf/1905.01164.pdf) for more details.


## Code

### Install dependencies

```
python -m pip install -r requirements.txt
```

This code was tested with python 3.6  

###  Train
To train SinGAN model on your own image, put the desire training image under Input/Images, and run

```
python main_train.py --input_wav_train <input_wav_file>
```

This will also use the resulting trained model to generate random samples starting from the coarsest scale (n=0).

To run this code on a cpu machine, specify `--not_cuda` when calling `main_train.py`

###  Paint to Image

To transfer a paint into a realistic image (See example in Fig. 11 in [the paper](https://arxiv.org/pdf/1905.01164.pdf), please first train SinGAN model on the desire image (as described above), then save your paint under "Input/Paint", and run the command

```
python paint2image.py --input_wav_train <training_wav_file> --input_wav_paint <paint_wav_file> --paint_start_scale <scale to inject>

```
Here as well, different injection scale will produce different editing effects. The coarsest injection scale equals 1. 

Advanced option: Specify quantization_flag to be True, to re-train *only* the injection level of the model, to get a on a color-quantized version of upsamled generated images from previous scale. For some images, this might lead to more realistic results.

## Use Case

We make use of the Paint2Image task of SinGAN to generate noisy speech samples from clean samples. The flow of the program is given as follows:

- Train SinGAN with Spectrogram of a Noisy sample
- Use Paint2Image task of the SinGAN model to generate samples at various scales.

### Note
SinGAN uses a multi-scale architecture that allows us to control the detail of the generated output. Read the [paper](https://arxiv.org/pdf/1905.01164.pdf) to understand; we take care of extracting the spectrogram, reconstructing the audio file internally. The end-user only has to deal with audio files.