# A Customized Denoiser for Your Own Camera!

- [A Customized Denoiser for Your Own Camera!](#a-customized-denoiser-for-your-own-camera)
  - [Data Collection](#data-collection)
    - [Explanation on `Ratio` and `Exposure Level`](#explanation-on-ratio-and-exposure-level)
  - [Fine-tuning!](#fine-tuning)
    - [Write Your Data Pair List](#write-your-data-pair-list)
    - [Fine-tuning!](#fine-tuning-1)
  - [Test Your Own Images!](#test-your-own-images)

## Data Collection

As stated in our main paper, in order to enable the denoiser to fully learn the linear relationship between ISO (overall system gain) and noise variance, we need two sets of images with significantly different ISO values (ISO $\le$ 500 and ISO $\ge$ 3200) for training.

Therefore, for each different digital gain (a.k.a. Ratio), we need to fine-tune our LED with two sets of different ISO data.

### Explanation on `Ratio` and `Exposure Level`

As for the digital gain, it can be calculted by dividing the exposure level of a well exposed image by the exposure level of the under-expisured image.<br/>
And the exposure value can be calculated by ISO * ExposureTime.

For example, the well-exposed image is shot on exposure time 30s and ISO 100, then the exposure level is 3000s. <br/>
And the under-exposed image is shot on exposure time 1/10s and ISO 6400, the exposure level is 640s. <br/>
So the ratio is 3000 / 640 = 4.6875.<br/>

To customize your denoiser, we need paired data of different ISOs at the same ratio. Therefore, in the aforementioned scenario, we also require an additional set of data with an exposure level of 640s (e.g., exposure time of 2s, ISO 320).

## Fine-tuning!

After you collected the paired images of your camera, it's time to fine-tune and get your own denoiser!

### Write Your Data Pair List

LED needs to know the specific correspondence of the paired data.
Therefore, you need to write a txt file to inform about the specific correspondence.

For example, we have collect the images `IMG_0001.CR2`, `IMG_0002.CR2`, `IMG_0003.CR2`, `IMG_0004.CR2`. <br/>
`IMG_0001.CR2`, `IMG_0002.CR2` is a corresponding pairs, and `IMG_0003.CR2`, `IMG_0004.CR2` is another pairs.

The data pair list should be like:
```txt
IMG_0001.CR2 IMG_0002.CR2
IMG_0003.CR2 IMG_0004.CR2
```

### Fine-tuning!

Fine-tuning for your denoiser with our official script!

```bash
python scripts/cutomized_denoiser.py -t [TAG] \
                                     -p [PRETRAINED_LED_MODEL] \
                                     --dataroot your/path/to/the/pairs \
                                     --data_pair_list your/path/to/the/txt
# Then the checkpoints can be found in experiments/[TAG]/models
# If you are a seasoned user of BasicSR, you can use "--force_yml" to further fine-tune the details of the options.
```

## Test Your Own Images!

We provide a script for testing your own RAW images in [image_process.py](/scripts/image_process.py). <br/>
You could run `python scripts/image_process.py --help` to get detailed information of this scripts.
> If your camera model is one of {Sony A7S2, Nikon D850}, you can found our pretrained model in [pretrained-models.md](/docs/pretrained-models.md).
>
> **Notice that**, if you wish to use the model from release v0.1.1, you need to add the `-opt` parameter: For NAFNet, add `-opt options/base/network_g/nafnet.yaml`. For Restormer, add `-opt options/base/network_g/restormer.yaml`.
```bash
usage: image_process.py [-h] -p PRETRAINED_NETWORK --data_path DATA_PATH [--save_path SAVE_PATH] [-opt NETWORK_OPTIONS] [--ratio RATIO] [--bps BPS] [--led]

optional arguments:
  -h, --help            show this help message and exit
  -p PRETRAINED_NETWORK, --pretrained_network PRETRAINED_NETWORK
                        the pretrained network path.
  --data_path DATA_PATH
                        the folder where contains only your raw images.
  --save_path SAVE_PATH
                        the folder where to save the processed images (in rgb), DEFAULT: 'inference/image_process'
  -opt NETWORK_OPTIONS, --network_options NETWORK_OPTIONS
                        the arch options of the pretrained network, DEFAULT: 'options/base/network_g/unet.yaml'
  --ratio RATIO, --dgain RATIO
                        the ratio/additional digital gain you would like to add on the image, DEFAULT: 1.0.
  --bps BPS, --output_bps BPS
                        the bit depth for the output png file, DEFAULT: 16.
  --led                 if you are using a checkpoint fine-tuned by our led.
```
