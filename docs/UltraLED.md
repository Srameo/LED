**Environment**

We employed [the same environment as LED](https://github.com/Srameo/LED/blob/main/docs/install.md), except for the rawpy library. For metric testing in our paper, we used the official rawpy library version 0.16.0 to ensure a fairer comparison against RGB-based methods.

Using the [rawpy](https://github.com/Vandermode/ELD#prerequisites) library improved by the ELD authors—which matches LED—yields better alignment and visualization results, consequently achieving different PSNR/SSIM/LPIPS metrics:

|  Ratio   |        ✖️50         |        ✖️100        |        ✖️200        |
| :------: | :----------------: | :----------------: | :----------------: |
|   LED    | 29.629/0.926/0.073 | 29.611/0.917/0.101 | 29.103/0.891/0.152 |
|   ELD    | 29.711/0.925/0.072 | 29.703/0.915/0.099 | 29.011/0.887/0.159 |
|    PG    | 28.742/0.921/0.091 | 28.322/0.903/0.132 | 27.254/0.855/0.208 |
| UltraLED | 31.271/0.939/0.066 | 31.126/0.929/0.094 | 30.351/0.898/0.139 |



**Inference**

1. Download the [test data](https://drive.google.com/file/d/1dL3MkhdK0xBnIEm71tBbDel2qUciOM5s/view?usp=sharing) and [ground truth](https://drive.google.com/file/d/1yPeYxC7_J8TsPg-MJqTi18ElUqX-t5qQ/view?usp=sharing).
2. Download the [pre-trained ratio map estimator model](https://drive.google.com/file/d/1226cS2Suj2uwqH3FdH6cdg50N-X-TO6Z/view?usp=sharing) and the [pre-trained raw denoiser model](https://drive.google.com/file/d/1tJulnDgFLCzScjTHvTnK2m8u-fC5XtFw/view?usp=sharing).

```python
python scripts/image_process_ultraled.py 

optional arguments:
  -p the pretrained ratio map estimator network path.
  -pd the pretrained network path for denoising.
  --data_path the folder containing raw images to be processed.
  --save_path output folder for processed images.
  -opt ratio map estimator network architecture options.
  -optd denoising network architecture options.
  --ratio maximum exposure gain ratio.
  [--target_exposure target exposure (overrides ratio).]
  [--bps output bit depth.]
```



**Evaluation**

Replace `folder_A` and `folder_B` in `scripts/test_metrics_ultraled.py` with the paths to the [test data](https://drive.google.com/file/d/1dL3MkhdK0xBnIEm71tBbDel2qUciOM5s/view?usp=sharing) and [ground truth](https://drive.google.com/file/d/1yPeYxC7_J8TsPg-MJqTi18ElUqX-t5qQ/view?usp=sharing), respectively.

```python
python scripts/test_metrics_ultraled.py 
```



**Train**

**Preparing Training Data**

We used the HDR subset from the [RAWHDR](https://github.com/yunhao-zou/RawHDR) dataset as our training data. We have converted this data into npz files following [the method described in LED](https://github.com/Srameo/LED/blob/main/docs/benchmark.md). You can directly download the processed npz training data using [Baidu Netdisk](https://pan.baidu.com/s/1wwm4dc2V7yKzvYXYYBZ4og?pwd=kngb) or [Google Drive](https://drive.google.com/file/d/1ir6QITP8mcSX8IFQCIkdO-luWdb0zCvj/view?usp=sharing) (note: we only used the HDR data, but we converted all data into npz format and provide their corresponding paired txt files).

To begin training, replace the `datasets/train/dataroot` parameter in `options/base/dataset/pretrain/UltraLED_trainS1.yaml` and ``options/base/dataset/pretrain/UltraLED_trainS2.yaml`` with the path to the downloaded npz data.



Step1

```python
python led/train.py -opt options/UltraLED/train_step1.yaml
```

Step2

Replace the `network_d_path` parameter in `options/UltraLED/train_step2.yaml` with the path to the model trained in Step 1.

```python
python led/train.py -opt options/UltraLED/train_step2.yaml
```

