# Model Zoo

- [Model Zoo](#model-zoo)
  - [Network for **Customizing Your Denoiser**!](#network-for-customizing-your-denoiser)
  - [Network for Benchmark](#network-for-benchmark)
  - [Noise Model](#noise-model)

## Network for **Customizing Your Denoiser**!

> The models provided in this section are meant for **customizing your own denoiser**! A summary of all the models will be available on Google Drive (we are working on it).<br/>
> You can find the detailed step-by-step process in the [demo.md](../docs/demo.md).

We are currently dedicated to training an exceptionally capable network that can generalize well to various scenarios using <strong>only two data pairs</strong>! We will update this section once we achieve our goal. Stay tuned and look forward to it!<br/>
Or you can just use the following pretrained LED module for custumizing on your own cameras!.

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
  <tr>
    <td>LED</td>
    <th> ELD (5 Virtual Cameras) </th>
    <th> Pretrain </th>
    <th> Restormer </th>
    <th> ELD </th>
    <th> 100-300 </th>
    <th> - </th>
    <th> - </th>
    <th> [<a href="https://drive.google.com/file/d/1iKNLaNRH5UejstaZbuq83yAdYxLaPa4x/view?usp=drive_link">Google Drive</a>] </th>
    <th> [<a href="/options/LED/other_arch/Restormer/LED+Restormer_Pretrain.yaml">options/LED/other_arch/Restormer/LED+Restormer_Pretrain.yaml</a>] </th>
  </tr>
  <tr>
    <td>LED</td>
    <th> ELD (5 Virtual Cameras) </th>
    <th> Pretrain </th>
    <th> NAFNet </th>
    <th> ELD </th>
    <th> 100-300 </th>
    <th> - </th>
    <th> - </th>
    <th> [<a href="https://drive.google.com/file/d/1FmqGv_YICLX4Gc-aWzvcTl8aWgeJ5cEB/view?usp=drive_link">Google Drive</a>] </th>
    <th> [<a href="/options/LED/other_arch/NAFNet/LED+NAFNet_Pretrain.yaml">options/LED/other_arch/NAFNet/LED+NAFNet_Pretrain.yaml</a>] </th>
  </tr>
</table>

## Network for Benchmark

> The models provided in this section are used to replicate the metrics in our paper, and you can find a summary of all the models on [Google Drive](https://drive.google.com/drive/folders/1UWZkeI_Aqdmy2U9vsoJyZLv33Ly__Art?usp=drive_link) or [Baidu Clould](https://pan.baidu.com/s/1fdG7zmsoyb_7j55RPHCICw?pwd=6bcz).
>
> Notice that, all the models are trained for bayer format.<br/>
> `Training Strategy` determines the training strategy we use, you can find the specific method in their paper ([ELD](https://openaccess.thecvf.com/content_CVPR_2020/papers/Wei_A_Physics-Based_Noise_Formation_Model_for_Extreme_Low-Light_Raw_Denoising_CVPR_2020_paper.pdf) and [PMN](https://arxiv.org/abs/2207.06103)).<br/>
> PMN* or ELD* for LED denotes the model is pretrained on that strategy.

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
  <tr>
    <td>LED</td>
    <th> Real Noise (6 Pairs on SID SonyA7S2) </th>
    <th> Finetune / Deploy </th>
    <th> UNet </th>
    <th> PMN* </th>
    <th> 100-300 </th>
    <th> SonyA7S2 </th>
    <th> SID SonyA7S2 </th>
    <th> [<a href="https://drive.google.com/file/d/1OocWgk6hENF3XwEDU3pah-3-NVGx_xJF/view?usp=drive_link">Google Drive</a>] </th>
    <th> [<a href="/options/LED/finetune/SID_SonyA7S2_MM22_PMN_Setting.yaml">options/LED/finetune/SID_SonyA7S2_MM22_PMN_Setting.yaml</a>] </th>
  </tr>
  <tr>
    <td>LED</td>
    <th> Real Noise (6 Pairs on SID SonyA7S2) </th>
    <th> Finetune / Deploy </th>
    <th> UNet </th>
    <th> ELD* </th>
    <th> 100-300 </th>
    <th> SonyA7S2 </th>
    <th> SID SonyA7S2 </th>
    <th> [<a href="https://drive.google.com/file/d/1vmeJBdXSjecnbTXLrMHOIhLgJbMFRaA1/view?usp=drive_link">Google Drive</a>] </th>
    <th> [<a href="/options/LED/finetune/SID_SonyA7S2_CVPR20_ELD_Setting.yaml">options/LED/finetune/SID_SonyA7S2_CVPR20_ELD_Setting.yaml</a>] </th>
  </tr>
  <tr>
    <td>LED</td>
    <th> Real Noise (6 Pairs on SID SonyA7S2) </th>
    <th> Finetune / Deploy </th>
    <th> Restormer </th>
    <th> ELD* </th>
    <th> 100-300 </th>
    <th> SonyA7S2 </th>
    <th> SID SonyA7S2 </th>
    <th> [<a href="https://drive.google.com/file/d/1KqY15BeXwjlwXGU5mEywBv3A2nbso7UL/view?usp=drive_link">Google Drive</a>] </th>
    <th> [<a href="/options/LED/other_arch/Restormer/LED+Restormer_Finetune.yaml">options/LED/other_arch/Restormer/LED+Restormer_Finetune.yaml</a>] </th>
  </tr>
  <tr>
    <td>LED</td>
    <th> Real Noise (6 Pairs on SID SonyA7S2) </th>
    <th> Finetune / Deploy </th>
    <th> NAFNet </th>
    <th> ELD* </th>
    <th> 100-300 </th>
    <th> SonyA7S2 </th>
    <th> SID SonyA7S2 </th>
    <th> [<a href="https://drive.google.com/file/d/11bgv3cD02ea0SU8mkL_W1UCNaH_FDbSS/view?usp=drive_link">Google Drive</a>] </th>
    <th> [<a href="/options/LED/other_arch/NAFNet/LED+NAFNet_Finetune.yaml">options/LED/other_arch/NAFNet/LED+NAFNet_Finetune.yaml</a>] </th>
  </tr>
  <tr>
    <td>LED</td>
    <th> Real Noise (24 Pairs on ELD SonyA7S2) </th>
    <th> Finetune / Deploy </th>
    <th> UNet </th>
    <th> ELD* </th>
    <th> 1-200 </th>
    <th> SonyA7S2 </th>
    <th> ELD SonyA7S2 </th>
    <th> [<a href="https://drive.google.com/file/d/1jpcMKqC59iVLmTMI8lCLQxd3wwg0dr76/view?usp=drive_link">Google Drive</a>] </th>
    <th> [<a href="/options/LED/finetune/ELD_SonyA7S2_CVPR20_ELD_Setting.yaml">options/LED/finetune/ELD_SonyA7S2_CVPR20_ELD_Setting.yaml</a>] </th>
  </tr>
  <tr>
    <td>LED</td>
    <th> Real Noise (24 Pairs on ELD NikonD850) </th>
    <th> Finetune / Deploy </th>
    <th> UNet </th>
    <th> ELD* </th>
    <th> 1-200 </th>
    <th> NikonD850 </th>
    <th> ELD NikonD850 </th>
    <th> [<a href="https://drive.google.com/file/d/1OScdkp6vXLe4sF__pMJhwnIqhnUCMcFo/view?usp=drive_link">Google Drive</a>] </th>
    <th> [<a href="/options/LED/finetune/ELD_NikonD850_CVPR20_ELD_Setting.yaml">options/LED/finetune/ELD_NikonD850_CVPR20_ELD_Setting.yaml</a>] </th>
  </tr>
  <tr>
    <td>ELD</td>
    <th> ELD (SonyA7S2) </th>
    <th> Deploy </th>
    <th> UNet </th>
    <th> PMN </th>
    <th> 100-300 </th>
    <th> SonyA7S2 </th>
    <th> SID SonyA7S2 </th>
    <th> [<a href="https://drive.google.com/file/d/1F4pNlro1egoACSgcXOLaARXx6ko35Gvi/view?usp=drive_link">Google Drive</a>] </th>
    <th> [<a href="/options/ELD/SID_SonyA7S2_MM22_PMN_Setting.yaml">options/ELD/SID_SonyA7S2_MM22_PMN_Setting.yaml</a>] </th>
  </tr>
  <tr>
    <td>ELD</td>
    <th> ELD (SonyA7S2) </th>
    <th> Deploy </th>
    <th> UNet </th>
    <th> ELD </th>
    <th> 100-300 </th>
    <th> SonyA7S2 </th>
    <th> SID SonyA7S2 </th>
    <th> [<a href="https://drive.google.com/file/d/1XecR4zXZOLxJqmDfK6WqnmPPUXi6YNkd/view?usp=drive_link">Google Drive</a>] </th>
    <th> [<a href="/options/ELD/SID_SonyA7S2_CVPR20_ELD_Setting.yaml">options/ELD/SID_SonyA7S2_CVPR20_ELD_Setting.yaml</a>] </th>
  </tr>
  <tr>
    <td>ELD</td>
    <th> ELD (SonyA7S2) </th>
    <th> Deploy </th>
    <th> UNet </th>
    <th> ELD </th>
    <th> 1-200 </th>
    <th> SonyA7S2 </th>
    <th> ELD SonyA7S2 </th>
    <th> [<a href="https://drive.google.com/file/d/1o0kr446Se2j5iemXiLjHxXy5UjLqgc6c/view?usp=drive_link">Google Drive</a>] </th>
    <th> [<a href="/options/ELD/ELD_SonyA7S2_CVPR20_ELD_Setting.yaml">options/ELD/ELD_SonyA7S2_CVPR20_ELD_Setting.yaml</a>] </th>
  </tr>
  <tr>
    <td>ELD</td>
    <th> ELD (NikonD850) </th>
    <th> Deploy </th>
    <th> UNet </th>
    <th> ELD </th>
    <th> 1-200 </th>
    <th> NikonD850 </th>
    <th> ELD NikonD850 </th>
    <th> [<a href="https://drive.google.com/file/d/1seTloGkpM2XbeeSwKSJHvq9khICnbfLv/view?usp=drive_link">Google Drive</a>] </th>
    <th> [<a href="/options/ELD/ELD_NikonD850_CVPR20_ELD_Setting.yaml">options/ELD/ELD_NikonD850_CVPR20_ELD_Setting.yaml</a>] </th>
  </tr>

  <tr>
    <td>P-G</td>
    <th> P-G (SonyA7S2) </th>
    <th> Deploy </th>
    <th> UNet </th>
    <th> ELD </th>
    <th> 100-300 </th>
    <th> SonyA7S2 </th>
    <th> SID SonyA7S2 </th>
    <th> [<a href="https://drive.google.com/file/d/1aNVsic0BRWPESUk6yU-OFcLYgCtdZkOj/view?usp=drive_link">Google Drive</a>] </th>
    <th> [<a href="/options/P-G/SID_SonyA7S2_CVPR20_ELD_Setting.yaml">options/P-G/SID_SonyA7S2_CVPR20_ELD_Setting.yaml</a>] </th>
  </tr>
  <tr>
    <td>P-G</td>
    <th> P-G (SonyA7S2) </th>
    <th> Deploy </th>
    <th> UNet </th>
    <th> ELD </th>
    <th> 1-200 </th>
    <th> SonyA7S2 </th>
    <th> ELD SonyA7S2 </th>
    <th> [<a href="https://drive.google.com/file/d/1B-92vP5RzK1xv8nLVnrIOb2IBbjHGJRU/view?usp=drive_link">Google Drive</a>] </th>
    <th> [<a href="/options/P-G/ELD_SonyA7S2_CVPR20_ELD_Setting.yaml">options/P-G/ELD_SonyA7S2_CVPR20_ELD_Setting.yaml</a>] </th>
  </tr>
  <tr>
    <td>P-G</td>
    <th> P-G (NikonD850) </th>
    <th> Deploy </th>
    <th> UNet </th>
    <th> ELD </th>
    <th> 1-200 </th>
    <th> NikonD850 </th>
    <th> ELD NikonD850 </th>
    <th> [<a href="https://drive.google.com/file/d/1m7l2EsaoZQoM-9ot2kFiTEQdDs5vYwi-/view?usp=drive_link">Google Drive</a>] </th>
    <th> [<a href="/options/P-G/ELD_NikonD850_CVPR20_ELD_Setting.yaml">options/P-G/ELD_NikonD850_CVPR20_ELD_Setting.yaml</a>] </th>
  </tr>
  <tr>
    <td>SID</td>
    <th> Real Noise (SonyA7S2) </th>
    <th> Deploy </th>
    <th> UNet </th>
    <th> ELD </th>
    <th> 100-300 </th>
    <th> SonyA7S2 </th>
    <th> SID SonyA7S2 </th>
    <th> [<a href="https://drive.google.com/file/d/1Rz5oShriEBFYQLiIXL8uKVhkWFsZuY9P/view?usp=drive_link">Google Drive</a>] </th>
    <th> [<a href="/options/SID/SID_SonyA7S2_CVPR_ELD_Setting.yaml">options/SID/SID_SonyA7S2_CVPR_ELD_Setting.yaml</a>] </th>
  </tr>
</tbody>
</table>


## Noise Model

> `Type` can be found in [`led/data/noise_utils/noise_generator.py`](../led/data/noise_utils/noise_generator.py).<br/>
> `p,g,t,r,q,c` in `Noise Type` denotes shot, read (gaussian), read (tukey-lambda), row, quant noise and black level error, respectively.<br/>
> All the noise model can be found in [Google Drive](https://drive.google.com/drive/folders/1newxmKSByfp2UyS1Hyvrtrs8UzP7WMiH?usp=drive_link) or [Baidu Cloud](https://pan.baidu.com/s/178PrumpQM-gb2Nfte3t4oA?pwd=t2a8).

<table>
<thead>
  <tr>
    <th> Type </th>
    <th> Noise Model </th>
    <th> Noise Type </th>
    <th> Camera Model </th>
    <th> :link: Download Links </th>
  </tr>
</thead>
<tbody>
  <tr>
    <td> VirtualNoisyPairGenerator </td>
    <th> ELD (5 Virtual Cameras) </th>
    <th> ptrqc </th>
    <th> Virtual Camera </th>
    <th> [<a href="https://drive.google.com/file/d/1XgL8ofYcD0LWGI0VpaG6LjtddmSoUL2j/view?usp=drive_link">Google Drive</a>] </th>
  </tr>
  <tr>
    <td> CalibratedNoisyPairGenerator </td>
    <th> ELD (SonyA7S2) </th>
    <th> ptrqc </th>
    <th> SonyA7S2 </th>
    <th> [<a href="https://drive.google.com/file/d/1XjzOKoqZ36CMc_cesuTQX8zQrOUHJHKS/view?usp=drive_link">Google Drive</a>] </th>
  </tr>
  <tr>
    <td> CalibratedNoisyPairGenerator </td>
    <th> ELD (NikonD850) </th>
    <th> ptrqc </th>
    <th> NikonD850 </th>
    <th> [<a href="https://drive.google.com/file/d/1XdDsm7jgZcSzqScrCCCnFlRiotERq1Jn/view?usp=drive_link">Google Drive</a>] </th>
  </tr>
  <tr>
    <td> CalibratedNoisyPairGenerator </td>
    <th> P-G (SonyA7S2) </th>
    <th> pg </th>
    <th> SonyA7S2 </th>
    <th> [<a href="https://drive.google.com/file/d/1HRvepo_AgB2fH7QMt11Kv89-wAuY9zh3/view?usp=drive_link">Google Drive</a>] </th>
  </tr>
  <tr>
    <td> CalibratedNoisyPairGenerator </td>
    <th> P-G (NikonD850) </th>
    <th> pg </th>
    <th> NikonD850 </th>
    <th> [<a href="https://drive.google.com/file/d/185hWUXjsmPSNr8LIQTTTZWdMKsTsLTzt/view?usp=drive_link">Google Drive</a>] </th>
  </tr>
</tbody>
</table>
