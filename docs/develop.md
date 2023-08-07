# Develop

> This repository borrows heavily from [BasicSR](https://github.com/XPixelGroup/BasicSR).
>
> If you are a beginner of BasicSR, please refer to [BasicSR](https://github.com/XPixelGroup/BasicSR).

- [Develop](#develop)
  - [More Than BasicSR!](#more-than-basicsr)
  - [Try RepNR Block on Different Architectures!](#try-repnr-block-on-different-architectures)
  - [Use the Noisy Pair Generator for Your Project.](#use-the-noisy-pair-generator-for-your-project)

## More Than BasicSR!

Compared with BasicSR, the option files could be more simple with a `base` key:

```yaml
base:
- options/base/dataset/pretrain/SID_raw_gt.yaml           # train dataset
- options/base/dataset/test/SID_SonyA7S2_val_split.yaml   # test dataset
- options/base/network_g/repnr_unet.yaml                  # network_g
- options/base/noise_g/noise_g_virtual.yaml               # noise_g
- options/base/pretrain/MM22_PMN.yaml                     # train
- options/base/val_and_logger.yaml                        # val + logger

name: LED_Pretrain_MM22_PMN_Setting
model_type: RAWImageDenoisingModel
scale: 1
num_gpu: 1
manual_seed: 2022

path:
  pretrain_network_g: ~
  predefined_noise_g: ~
  strict_load_g: true
  resume_state: ~
  CRF: datasets/ICCV23-LED/EMoR

val:
  illumination_correct: true
```

Just use the relative path to project root directory for a cleaner and simpler config files!


## Try RepNR Block on Different Architectures!

> Build a new architecture with RepNR block using only **a line of code**!

We provide a efficient architecture converter for your any architecture with `torch.nn.Conv2d`. This converter could automiticly replace the `torch.nn.Conv2d` with our `RepNR` block.

```python
from led.archs.repnr_utils import build_repnr_arch_from_base

repnr_opt = {
    'dont_convert_module': ['conv10_1'],   # conv10_1 will not be convert into repnr block
    'branch_num': 5,
    'align_opts': {
        'init_weight': 1.0,
        'init_bias': 0.0
    },
    'aux_conv_opts': {
        'bias': True,
        'init': 'zero_init_'
    }
}
my_own_net = MyOwnArch()

# build repnr block from your defined architecture!
repnr_arch = build_repnr_arch_from_base(my_own_net, repnr_opt)
```

Also, you could simply convert the repnr_arch back into your module by:
```python
# Averaging the CSAs.
repnr_arch.generalize()

# The weight will be reparameteraztion.
my_own_net = repnr_arch.deploy()
```

## Use the Noisy Pair Generator for Your Project.

We provide two kinds of noisy pair generator (calibration or with virtual cameras) in `led/data/noise_utils/noise_generator.py`.
