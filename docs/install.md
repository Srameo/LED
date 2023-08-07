# INSTALLATION

Installation is tested and working on the following platforms:

- Ubuntu 20.04.1
  - GPUs: RTX-3090 (Driver Version: 520.56.06)
  - RAM: 64GB

## Prerequisites

The detailed package requirement can be found in [`requirements.txt`](../requirements.txt).<br/>
To install the packages required by our LED, you could :

1. Create a virtual enviorment and activate it:
   ```bash
   conda create -y -n LED-ICCV23 python=3.8
   conda activate LED-ICCV23
   ```
2. Install the prerequisite packages:
   ```bash
   pip install -r requirements.txt
   ```
3. Install our LED for develop:
   ```bash
   python setup.py develop
   ```

**Attention!** ONE package ([**`RawPy`**](https://github.com/letmaik/rawpy)) is not included in requirements.txt! <br/>
To read the `cam2rgb` matrix from RAW format data, we use the customized rawpy by [Vandermode](https://github.com/Vandermode) (the author of [ELD](https://github.com/Vandermode/ELD)) during data preparation.
> By the way, if you don't need to reproduce the metrics in our paper, you can simply install rawpy through `pip install rawpy` and skip the following part. But in this way, you won't be able to use the dataloader we have already prepared.

We heavily recommend you follow the instructions from [ELD](https://github.com/Vandermode/ELD#prerequisites) to install the custumized rawpy, or you can just follow the next steps.

1. Download custumized rawpy and LibRaw, then unzip them:
   ```bash
   # download in downloads/
   mkdir -p downloads/
   # use our script for downloading from google drive
   python scripts/download_gdrive.py --id 1B4gJYe3h4UxWTMZzNL2xVodSDUlNbeAR --save-path downloads/LibRaw-0.19.1.zip
   python scripts/download_gdrive.py --id 1EuJsbZ_a_YJHHcGAVA9TXXPnGU90QoP4 --save-path downloads/rawpy.zip
   # unzip the rawpy and LibRaw
   unzip downloads/LibRaw-0.19.1.zip -d downloads/
   unzip downloads/rawpy.zip -d downloads/
   ```
2. Compile and install LibRaw:
   ```bash
   cd downloads/LibRaw-0.19.1
   ./configure
   make
   ```
3. Install RawPy! (Please pay attention to whether you are in a virtual environment):
   ```bash
   cd ../rawpy
   pip install -e .
   ```

All the above instructions are integrated into install.sh.<br/>
So you can simply install by `bash install.sh`.

## Pretrained Models

If you would like to use our pretrained network, please refer to [pretrained-models.md](pretrained-models.md).
