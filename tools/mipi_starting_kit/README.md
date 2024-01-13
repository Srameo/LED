<p align="center">
  <img src='/.assets/logo-mipi.svg' alt='ICCV23_LED_LOGO' width='200px'/><br/>
</p>

# Starting-kit for Few-shot RAW Image Denosing @MIPI2024

\[[Homepage](https://mipi-challenge.org/MIPI2024/)\] \[[Codalab](https://codalab.lisn.upsaclay.fr/competitions/17017)\]

- [Starting-kit for Few-shot RAW Image Denosing @MIPI2024](#starting-kit-for-few-shot-raw-image-denosing-mipi2024)
  - [Overview](#overview)
  - [Tips](#tips)
  - [Possible Solution](#possible-solution)
  - [Related Dataset](#related-dataset)


## Overview

The Few-shot RAW Image Denoising track is geared towards training neural networks for raw image denoising in scenarios where paired data is limited.

In this starting kit, we will provide you with a possible solution, but you don't have to follow this approach.

Additionally, we will also provide you with tips on important considerations during the competition and the submission process.

In the [`code_example/tutorial.ipynb`](/tools/mipi_starting_kit/code_example/tutorial.ipynb), we provide examples and notes on reading data, lite ISP, calculating scores, and submission.

In the [`evaluate`](/tools/mipi_starting_kit/evaluate), we provide the validation code that we submitted on Codalab.

## Tips

- You are **NOT** restricted to train their algorithms only on the provided dataset. Other **PUBLIC** dataset
  can be used as well. However, you need to mention in the final submitted factsheet what public datasets you have used.
- For different cameras, you can test using different neural network weights.
- Please ensure that your testing process can be conducted on a single NVIDIA RTX 3090 (i.e., the memory usage needs to be less than 24GB). This is to limit resource usage during deployment.
-  We will check the participants' code after the final test stage to ensure fairness.

## Possible Solution

A viable solution is to train following the pre-train and fine-tune strategy in LED.

During the pre-train phase, you can use other public datasets, and ultimately fine-tune on the data we provide.

We offer a config file in [`option_example/finetune.yaml`](/tools/mipi_starting_kit/option_example/finetune.yaml) that can be used with the LED codebase.

Of course, you are **NOT** restricted to using this approach.

## Related Dataset

The section where we provide relevant datasets serves two purposes:

- You can use the clean RAW images included for data synthesis and pre-training of neural networks.
- Since the provided data only includes training data, we recommend that you first test your algorithm on the following two datasets. Afterwards, choose a finalized approach to perform validation/test on Codalab.

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
