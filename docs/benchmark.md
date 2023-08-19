# Training and Evaluation

> For training and evaluation, you **must** install the customized `rawpy` package provided in ELD. <br/>
> If you have installed the environment using our `install.sh` script, you can ignore this message. <br/>
> Otherwise, please refer to [install.md](install.md) for further instructions.

- [Training and Evaluation](#training-and-evaluation)
  - [Data Preparation](#data-preparation)
    - [Acceleration for Training and Testing](#acceleration-for-training-and-testing)
    - [Download the Data Pair List](#download-the-data-pair-list)
    - [Visualization](#visualization)
  - [Pretrained Models](#pretrained-models)
  - [Training](#training)
  - [Evaluation](#evaluation)

## Data Preparation

<table>
<thead>
  <tr>
    <th> Dataset </th>
    <th> :link: Source </th>
    <th> Conf. </th>
    <th> Shot on </th>
    <th> CFA Pattern </th>
  </tr>
</thead>
<tbody>
  <tr>
    <td> SID </td>
    <th> [<a href='https://cchen156.github.io/SID.html'>Homepage</a>][<a href='https://github.com/cchen156/Learning-to-See-in-the-Dark'>Github</a>][Dataset (<a href='https://drive.google.com/file/d/1G6VruemZtpOyHjOC5N8Ww3ftVXOydSXx/view'>Google Drive</a> / <a href='https://pan.baidu.com/s/1fk8EibhBe_M1qG0ax9LQZA#list/path=%2F'>Baidu Clould</a>)] </th>
    <th> CVPR 2018 </th>
    <th> Sony A7S2 </th>
    <th> Bayer (RGGB) </th>
  </tr>
  <tr>
    <td> ELD </td>
    <th> [<a href='https://github.com/Vandermode/ELD'>Github</a>][<a href='https://drive.google.com/drive/folders/1QoEhB1P-hNzAc4cRb7RdzyEKktexPVgy'>Google Drive</a>][Dataset (<a href='https://drive.google.com/file/d/13Ge6-FY9RMPrvGiPvw7O4KS3LNfUXqEX/view?usp=drive_link'>Google Drive</a> / <a href='https://pan.baidu.com/share/init?surl=1ksugpPH5uyDL-Z6S71Q5g?pwd=0lby'>Baidu Clould</a>)] </th>
    <th> CVPR 2020 </th>
    <th> Sony A7S2 / Nikon D850 / Canon EOS70D / Canon EOS700D </th>
    <th> Bayer </th>
  </tr>
</tbody>
</table>

After download all the above datasets, you could symbol link them to the dataset folder.
```bash
mkdir datasets/ICCV23-LED && cd datasets/ICCV23-LED
ln -s your/path/to/SID/Sony  ./Sony
ln -s your/path/to/ELD       ./ELD_sym
```
Or just put them directly in the `datasets/ICCV23-LED` folder.

### Acceleration for Training and Testing

> If you have symbol link the data into `datasets/ICCV23-LED` like the aforementioned steps, you could just run
> ```bash
> bash scripts/data_preparation/accelerate.sh
> ```
> for all the data acceleration.
>
> To figure out what the script is actually doing, please continue reading.

**Training**

Like all other method, we use the `long` subset of SID Sony dataset for training. For fast trainine, we crop the data in advance.

```bash
# Extract patch for training.
python scripts/data_preparation/extract_bayer_subimages_with_metadata.py \
    --data-path datasets/ICCV23-LED/Sony/long \
    --save-path datasets/ICCV23-LED/Sony_train_long_patches \
    --suffix ARW \
    --n-thread 10
```

**Testing**

First, we convert the ELD data into SID data structure for use the same `dataset` class.

```bash
# Convert the ELD data into SID data structure
python scripts/data_preparation/eld_to_sid_structure.py \
    --data-path datasets/ICCV23-LED/ELD_sym
    --save-path datasets/ICCV23-LED/ELD
```

Then for testing, we convert all the RAW files into numpy array format.

```bash
# convert SID SonyA7S2
python scripts/data_preparation/bayer_to_npy.py --data-path datasets/ICCV23-LED/Sony --save-path datasets/ICCV23-LED/Sony_npy --suffix ARW --n-thread 8
# convert ELD SonyA7S2
python scripts/data_preparation/bayer_to_npy.py --data-path datasets/ICCV23-LED/ELD/SonyA7S2 --save-path datasets/ICCV23-LED/ELD_npy/SonyA7S2 --suffix ARW --n-thread 8
# convert ELD NikonD850
python scripts/data_preparation/bayer_to_npy.py --data-path datasets/ICCV23-LED/ELD/NikonD850 --save-path datasets/ICCV23-LED/ELD_npy/NikonD850 --suffix nef --n-thread 8
# convert ELD CanonEOS70D
python scripts/data_preparation/bayer_to_npy.py --data-path datasets/ICCV23-LED/ELD/CanonEOS70D --save-path datasets/ICCV23-LED/ELD_npy/CanonEOS70D --suffix CR2 --n-thread 8
# convert ELD CanonEOS700D
python scripts/data_preparation/bayer_to_npy.py --data-path datasets/ICCV23-LED/ELD/CanonEOS700D --save-path datasets/ICCV23-LED/ELD_npy/CanonEOS700D --suffix CR2 --n-thread 8
```

### Download the Data Pair List

> The summay of the data pair list can be found in [Google Drive](https://drive.google.com/drive/folders/1xZbJPfJoXmq4fWJWy3tXtULEn79Xoz5O?usp=drive_link).

> Since commit [`fadffc7`](https://github.com/Srameo/LED/commit/fadffc7282b02ab2fcc7fbade65f87217b642588), the data pair list for benchmark has been added in [datasets/txtfiles](../datasets/txtfiles).


Like SID, we use txt files to identify the images for training or testing.<br/>
To evalute or train LED using our proposed code, you should download the corresponding txt file and put them in the right place. Or change the `data_pair_list` property in `dataset:train:dataroot` option.
<!--
<table>
<thead>
  <tr>
    <th> Dataset </th>
    <th> :link: Source </th>
    <th> Phase </th>
    <th> Put in </th>
  </tr>
</thead>
<tbody>
  <tr>
    <td> SID SonyA7S2 </td>
    <th> [<a href="https://drive.google.com/drive/folders/1GLlrsfO0NqocI4vqn4YtQvpOreWYC_dT?usp=drive_link">Google Drive</a>] </th>
    <th> Train / Fine-tune / Test </th>
    <th> datasets/ICCV23-LED/Sony_npy </th>
  </tr>
  <tr>
    <td> ELD SonyA7S2 </td>
    <th> [<a href='https://drive.google.com/drive/folders/1ybdaACLApX3cLCmoQny45PAyMQ1nOVKO?usp=drive_link'>Google Drive</a>] </th>
    <th> Fine-tune / Test </th>
    <th> datasets/ICCV23-LED/ELD_npy/SonyA7S2 </th>
  </tr>
  <tr>
    <td> ELD NikonD850 </td>
    <th> [<a href='https://drive.google.com/drive/folders/19a4pUWiGA7xw7ssNaXMNHThL4Xv8sUfF?usp=drive_link'>Google Drive</a>] </th>
    <th> Fine-tune / Test </th>
    <th> datasets/ICCV23-LED/ELD_npy/NikonD850 </th>
  </tr>
  <tr>
    <td> ELD CanonEOS70D </td>
    <th> [<a href='https://drive.google.com/drive/folders/1KtBruEqekIgVHEi9X9c84Lvbsw5QZ123?usp=drive_link'>Google Drive</a>] </th>
    <th> Fine-tune / Test </th>
    <th> datasets/ICCV23-LED/ELD_npy/CanonEOS70D </th>
  </tr>
  <tr>
    <td> ELD CanonEOS700D </td>
    <th> [<a href='https://drive.google.com/drive/folders/1EopUTStJBAG1UgA4sTWGsFonkTex8jL6?usp=drive_link'>Google Drive</a>] </th>
    <th> Fine-tune / Test </th>
    <th> datasets/ICCV23-LED/ELD_npy/CanonEOS700D </th>
  </tr>
</tbody>
</table>
 -->

### Visualization

> Since commit [`fadffc7`](https://github.com/Srameo/LED/commit/fadffc7282b02ab2fcc7fbade65f87217b642588), the EMoR data for fast visualization has been added in [datasets/EMoR](../datasets/EMoR).

Download the EMoR files calibrated by [ELD](https://github.com/Vandermode/ELD) in [Google Drive](https://drive.google.com/drive/folders/1U6W-qXqnZl-5-dLpFhLAGLjniBH5yAYY?usp=drive_link) or [Baidu Clould](https://pan.baidu.com/s/1YW5yPTloDawasTrWlUWw0Q?pwd=y83f) for fast visualization using GPU.


## Pretrained Models

We provide a [model zoo](/docs/pretrained-models.md) for reproduce our LED. Please refer to `#Network for Benchmark` section in [pretrained-models.md](/docs/pretrained-models.md).


## Training

We have provided abundant configs for you to reproduce most of the metrics in our paper!
Just select a config and run:
```bash
python led/train.py -opt [OPT]
```
To learn more about the config, please move to [develop.md](/docs/develop.md).

## Evaluation

We have provided a script for fast evaluation:
```bash
python scripts/benckmark.py \
    -t [TAG] \
    -p [PRETRAINED_NET] \
    --dataset [DATASET] [CAMERA_MODEL] \
    [--led] \         # If the model is fine-tuned and deployed by our LED method.
    [--save_image] \  # If you would like to save the result

# e.g.
python scripts/benckmark.py \
    -t test \
    -p pretrained/network_g/LED_Deploy_SID_SonyA7S2_CVPR20_Setting_Ratio100-300.pth \
    --dataset SID SonyA7S2 \
    --led --save_image
# the log and visualization can be found in `results/test`
```
