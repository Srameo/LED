## We do not provide the download scripts after commit `eee99fbe23`!

Since our checkpoint cannot pass Google Drive's virus check, we no longer provide scripts for automatic downloading from Google Drive (which would result in a Google warning instead of the checkpoint).

![Google Warning](https://private-user-images.githubusercontent.com/23457638/290250935-9c881825-6add-4953-9969-385a6491ff83.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3MDQ1NDcyMTAsIm5iZiI6MTcwNDU0NjkxMCwicGF0aCI6Ii8yMzQ1NzYzOC8yOTAyNTA5MzUtOWM4ODE4MjUtNmFkZC00OTUzLTk5NjktMzg1YTY0OTFmZjgzLnBuZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNDAxMDYlMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjQwMTA2VDEzMTUxMFomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPTMwNDM2MmQzOWNkMjIxYzZkM2ZhZTlhNmZlODVhYjgwMDRlYzY3YjUyZWViMWViY2M4YjBlODQ2NjBkNzEyNzMmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0JmFjdG9yX2lkPTAma2V5X2lkPTAmcmVwb19pZD0wIn0.RCAms-LnGQQPoGr3b4HUvBljtnd55NItoLKoWh3c-ko)

For details, please refer to [issue#18](https://github.com/Srameo/LED/issues/18).

However, you could not only download the checkpoint from [Google Drive](https://drive.google.com/drive/folders/11MYkjzbPIZ7mJbu9vrgaVC-OwGcOFKsM?usp=sharing) but our [Github Release](https://github.com/Srameo/LED/releases/tag/0.1.0) as well.

## Explanation on the pretrained models

The data naming convention consists of the following components: `{Method}_{Phase}_{Dataset}_{Camera Model}_{Training Setting}_Setting_Ratio{Range}`.

For the LED method, there are two phases in total: `Pretrain` and `Deploy`. For `Pretrain`, the checkpoint obtained cannot be directly tested on any dataset; it is only used for subsequent fine-tuning. On the other hand, for all methods, the `Deploy` means that the checkpoint contains parameters consistent with the UNet used in the [SID](https://cchen156.github.io/SID.html) method, making it directly suitable for testing.

Regarding the training setting, there are two mainstream settings known as [ELD (CVPR20)](https://github.com/Vandermode/ELD) and [PMN (MM22)](https://github.com/megvii-research/PMN) settings. We represent these two settings as "CVPR20" and "MM22," respectively.

e.g. `LED_Deploy_SID_SonyA7S2_CVPR20_Setting_Ratio100-300`:
1. "LED_Deploy": This refers to the LED method in the "deploy" phase.
2. "SID_SonyA7S2": This indicates that the testing is done on the SonyA7S2 subset of the SID dataset.
3. "CVPR20_Setting": This means that the training strategy during the "pretrain" phase is the same as the one used in the "ELD (CVPR20)" setting.
4. "Ratio100-300": This indicates the range of the ratio is from 100 to 300.

## Explanation on the noisy pair generator

The data naming convention consists of the following components: `{Type}_{Noise Model}_{Noise Type}_{Camera Model}`.

e.g. `VirtualNoisyPairGenerator_ELD_ptrqc_5VirtualCameras.pth` denotes the `VirtualNoisyPairGenerator` with `ELD` noise model and shot (poission), read (tukey lambda), row, quant noise and color bias (black level error). Also the checkpoint contains 5 random sampled virtual cameras.