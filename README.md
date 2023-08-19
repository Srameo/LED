<p align="center">
  <img src='.assets/logo.svg' alt='ICCV23_LED_LOGO' width='200px'/>
</p>

# :bulb: LED: Lighting Every Darkness in Two Pairs!

This repository contains the official implementation of the following paper:
> Lighting Every Darkness in Two Pairs: A Calibration-Free Pipeline for RAW Denoising<br/>
> [Xin Jin](https://srameo.github.io)<sup>\*</sup>, [Jia-Wen Xiao](https://github.com/schuy1er)<sup>\*</sup>, [Ling-Hao Han](https://scholar.google.com/citations?user=0ooNdgUAAAAJ&hl=en), [Chunle Guo](https://mmcheng.net/clguo/), [Ruixun Zhang](https://www.math.pku.edu.cn/teachers/ZhangRuixun%20/index.html), [Xialei Liu](https://mmcheng.net/xliu/), [Chongyi Li](https://li-chongyi.github.io/)<br/>
> (\* denotes equal contribution.)<br/>
> In ICCV 2023

\[[Homepage](https://srameo.github.io/projects/led-iccv23/)\]
\[[Paper](https://arxiv.org/abs/2308.03448)]
\[[Google Drive](https://drive.google.com/drive/folders/11MYkjzbPIZ7mJbu9vrgaVC-OwGcOFKsM?usp=sharing) / [Baidu Clould](https://pan.baidu.com/s/17rA_8GvfNPZJY5Zl9dyILw?pwd=iay5)]
\[[知乎](https://zhuanlan.zhihu.com/p/648242095)\]
\[Poster (TBD)\]
\[Video (TBD)\]

<details>
<summary>Comparaison with Calibration-Based Method</summary>
<img src='https://github.com/Srameo/LED/assets/51229295/022505b0-8ff0-445b-ab1f-bb79b48ecdbd' alt='ICCV23_LED_TEASER0' width='500px'/>
</details>

LED is a **Calibration-Free** Pipeline for RAW Denoising (currently for extremely low-light conditions).

So tired of calibrating the noise model? Try our LED!<br/>
Achieveing <b style='font-size: large'>SOTA performance</b> in <b style='font-size: large'>2 paired data</b> and <b style='font-size: large'>training time less than 4mins</b>!

<table>
  <tbody>
    <tr><td><img src='https://github.com/Srameo/LED/assets/51229295/5311798d-f988-48f7-b50e-7cd080d7316c' alt='ICCV23_LED_TEASER1'/>
    </td><td><img src='https://github.com/Srameo/LED/assets/51229295/3403a346-cd54-435c-b0b3-46b716863719' alt='ICCV23_LED_TEASER2'/></td></tr>
    <tr><td><details><summary>More Teaser</summary><img src='https://github.com/Srameo/LED/assets/51229295/0c737715-919a-49a9-a115-76935b74a5bb' alt='ICCV23_LED_TEASER3'/></details></td>
    <td><details><summary>More Teaser</summary><img src='https://github.com/Srameo/LED/assets/51229295/c3af68de-9e6d-47c9-8365-743be671ad77' alt='ICCV23_LED_TEASER4'/></details></td></tr>
  </tbody>
</table>

- First of all, [:wrench: Dependencies and Installation](#wrench-dependencies-and-installation).
- For **academic research**, please refer to [pretrained-models.md](docs/pretrained-models.md) and [:robot: Training and Evaluation](#robot-training-and-evaluation).
- For **further development**, please refer to [:construction: Further Development](#construction-further-development).
- For **using LED on your own camera**, please refer to [:sparkles: Pretrained Models](#sparkles-pretrained-models) and [:camera: Quick Demo](#camera-quick-demo).

## :newspaper: News

> Future work can be found in [todo.md](docs/todo.md).

- **Aug 19, 2023**: Release relevent files on [Baidu Clould](https://pan.baidu.com/s/17rA_8GvfNPZJY5Zl9dyILw?pwd=iay5).
- **Aug 15, 2023**: For faster benchmark, we released the relevant files in commit [`fadffc7`](https://github.com/Srameo/LED/commit/fadffc7282b02ab2fcc7fbade65f87217b642588).
- **Aug, 2023**: We released a Chinese explanation of our paper on [知乎](https://zhuanlan.zhihu.com/p/648242095).
- **Aug, 2023**: Our code is publicly available!
- **July, 2023**: Our paper "Lighting Every Darkness in Two Pairs: A Calibration-Free Pipeline for RAW Denoising" has been accepted by ICCV 2023.


## :wrench: Dependencies and Installation

1. Clone and enter the repo:
   ```bash
   git clone https://github.com/Srameo/LED.git ICCV23-LED
   cd ICCV23-LED
   ```
2. Simply run the `install.sh` for installation! Or refer to [install.md](docs/install.md) for more details.
   > We use the customized rawpy package in [ELD](https://github.com/Vandermode/ELD), if you don't want to use it or want to know more information, please move to [install.md](docs/install.md)
   ```bash
   bash install.sh
   ```
3. Activate your env and start testing!
   ```bash
   conda activate LED-ICCV23
   ```

## :sparkles: Pretrained Models
> If your requirement is for **academic research** and you would like to benchmark our method, please refer to [pretrained-models.md](docs/pretrained-models.md), where we have a rich variety of models available across a diverse range of methods, training strategies, pre-training, and fine-tuning models.

We are currently dedicated to training an exceptionally capable network that can generalize well to various scenarios using <strong>only two data pairs</strong>! We will update this section once we achieve our goal. Stay tuned and look forward to it!<br/>
Or you can just use the following pretrained LED module for custumizing on your own cameras! (please follow the instruction in [Quick Demo](#quick-demo)).

<table>
<thead>
  <tr>
    <th> Method </th>
    <th> Noise Model </th>
    <th> Phase </th>
    <th> Framework </th>
    <th> Training Strategy </th>
    <th> Additional Dgain (ratio) </th>
    <th> Camera Model </th>
    <th> Validation on </th>
    <th> :link: Download Links </th>
    <th> Config File </th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>LED</td>
    <th> ELD (5 Virtual Cameras) </th>
    <th> Pretrain </th>
    <th> UNet </th>
    <th> PMN </th>
    <th> 100-300 </th>
    <th> - </th>
    <th> - </th>
    <th> [<a href="https://drive.google.com/file/d/1FSXp_vJxbo8_dbMJPiA33DZfagn1ExHA/view?usp=drive_link">Google Drive</a>] </th>
    <th> [<a href="/options/LED/pretrain/MM22_PMN_Setting.yaml">options/LED/pretrain/MM22_PMN_Setting.yaml</a>] </th>
  </tr>
  <tr>
    <td>LED</td>
    <th> ELD (5 Virtual Cameras) </th>
    <th> Pretrain </th>
    <th> UNet </th>
    <th> ELD </th>
    <th> 100-300 </th>
    <th> - </th>
    <th> - </th>
    <th> [<a href="https://drive.google.com/file/d/1kIN_eyNd4mlKhPV4PMmgzaoE3ddagjNU/view?usp=drive_link">Google Drive</a>] </th>
    <th> [<a href="/options/LED/pretrain/CVPR20_ELD_Setting.yaml">options/LED/pretrain/CVPR20_ELD_Setting.yaml</a>] </th>
  </tr>
  <tr>
    <td>LED</td>
    <th> ELD (5 Virtual Cameras) </th>
    <th> Pretrain </th>
    <th> UNet </th>
    <th> ELD </th>
    <th> 1-200 </th>
    <th> - </th>
    <th> - </th>
    <th> [<a href="https://drive.google.com/file/d/1IzOkJuHWQVXmkzFJzQ9-gkPXBlrutO2p/view?usp=drive_link">Google Drive</a>] </th>
    <th> [<a href="/options/LED/pretrain/CVPR20_ELD_Setting_Ratio1-200.yaml">options/LED/pretrain/CVPR20_ELD_Setting_Ratio1-200.yaml</a>] </th>
  </tr>
</table>

## :camera: Quick Demo

### Get Clean Images in the Dark!

We provide a script for testing your own RAW images in [image_process.py](scripts/image_process.py). <br/>
You could run `python scripts/image_process.py --help` to get detailed information of this scripts.
> If your camera model is one of {Sony A7S2, Nikon D850}, you can found our pretrained model in [pretrained-models.md](docs/pretrained-models.md).
```bash
usage: image_process.py [-h] -p PRETRAINED_NETWORK --data_path DATA_PATH [--save_path SAVE_PATH] [-opt NETWORK_OPTIONS] [--ratio RATIO] [--target_exposure TARGET_EXPOSURE] [--bps BPS] [--led]

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
  --target_exposure TARGET_EXPOSURE
                        Target exposure, activate this will deactivate ratio.
  --bps BPS, --output_bps BPS
                        the bit depth for the output png file, DEFAULT: 16.
  --led                 if you are using a checkpoint fine-tuned by our led.
```

### Fine-tune for Your Own Camera!

1. Collect noisy-clean image pairs for your camera model, please follow the insruction in [demo.md](docs/demo.md).
2. Select a **LED Pretrained** model in our [model zoo](docs/pretrained-models.md) (based on the additional dgain you want to add on the image), and fine-tune it using your data!
   ```bash
   python scripts/cutomized_denoiser.py -t [TAG] \
                                        -p [PRETRAINED_LED_MODEL] \
                                        --dataroot your/path/to/the/pairs \
                                        --data_pair_list your/path/to/the/txt
   # Then the checkpoints can be found in experiments/[TAG]/models
   # If you are a seasoned user of BasicSR, you can use "--force_yml" to further fine-tune the details of the options.
   ```
3. Get ready and test your denoiser! (move to [Get Clean Images in the Dark!](#get-clean-images-in-the-dark)).

## :robot: Training and Evaluation

Please refer to [benchmark.md](docs/benchmark.md) to learn how to benchmark LED, how to train a new model from scratch.

## :construction: Further Development

If you would like to develop/use LED in your projects, welcome to let us know. We will list your projects in this repository.<br/>
Also, we provide useful tools for your futher development, please refer to [develop.md](docs/develop.md).


## :book: Citation

If you find our repo useful for your research, please consider citing our paper:

```bibtex
@inproceedings{jiniccv23led,
    title={Lighting Every Darkness in Two Pairs: A Calibration-Free Pipeline for RAW Denoising},
    author={Jin, Xin and Xiao, Jia-Wen and Han, Ling-Hao and Guo, Chunle and Zhang, Ruixun and Liu, Xialei and Li, Chongyi},
    journal={Proceedings of the IEEE/CVF International Conference on Computer Vision},
    year={2023}
}
```

## :scroll: License

This code is licensed under the [Creative Commons Attribution-NonCommercial 4.0 International](https://creativecommons.org/licenses/by-nc/4.0/) for non-commercial use only.
Please note that any commercial use of this code requires formal permission prior to use.

## :postbox: Contact

For technical questions, please contact `xjin[AT]mail.nankai.edu.cn` and `xiaojw[AT]mail.nankai.edu.cn`.

For commercial licensing, please contact `cmm[AT]nankai.edu.cn`.

## :handshake: Acknowledgement

This repository borrows heavily from [BasicSR](https://github.com/XPixelGroup/BasicSR), [Learning-to-See-in-the-Dark](https://github.com/cchen156/Learning-to-See-in-the-Dark) and [ELD](https://github.com/Vandermode/ELD).<br/>
We would like to extend heartfelt gratitude to [Ms. Li Xinru](https://issuu.com/lerryn) for crafting the exquisite logo for our project.

<!-- We also thank all of our contributors.

<a href="https://github.com/Srameo/LED/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=Srameo/LED" />
</a> -->
