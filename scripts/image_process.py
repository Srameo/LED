from led.archs import build_network
from led.utils.options import yaml_load
from led.data.raw_utils import metainfo, pack_raw_bayer, depack_raw_bayer
import rawpy
import torch
from copy import deepcopy
import argparse
import glob
import numpy as np
import cv2
import os
from tqdm import tqdm

def load_network(net, load_path, strict=True, param_key='params'):
    """Load network.

    Args:
        load_path (str): The path of networks to be loaded.
        net (nn.Module): Network.
        strict (bool): Whether strictly loaded.
        param_key (str): The parameter key of loaded network. If set to
            None, use the root 'path'.
            Default: 'params'.
    """
    load_net = torch.load(load_path, map_location=lambda storage, loc: storage)
    if param_key is not None:
        if param_key not in load_net and 'params' in load_net:
            param_key = 'params'
            print('Loading: params_ema does not exist, use params.')
        load_net = load_net[param_key]
    print(f'Loading {net.__class__.__name__} model from {load_path}, with param key: [{param_key}].')
    # remove unnecessary 'module.'
    for k, v in deepcopy(load_net).items():
        if k.startswith('module.'):
            load_net[k[7:]] = v
            load_net.pop(k)
    net.load_state_dict(load_net, strict=strict)


def get_available_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def read_img(raw_path):
    raw = rawpy.imread(raw_path)
    raw_vis = raw.raw_image_visible.copy()
    raw_pattern = raw.raw_pattern
    black_level = np.array(raw.black_level_per_channel, dtype=np.float32).reshape(1, 4, 1, 1)
    white_level = np.array(raw.camera_white_level_per_channel, dtype=np.float32)
    if (white_level == None).any():
        white_level = np.array(raw.white_level, dtype=np.float32)
    if white_level.size == 1:
        white_level = white_level.repeat(4, 0)
    white_level = white_level.reshape(1, 4, 1, 1)
    raw_packed = torch.from_numpy(np.float32(pack_raw_bayer(raw_vis, raw_pattern))[np.newaxis]).contiguous()
    black_level = torch.from_numpy(black_level).contiguous()
    white_level = torch.from_numpy(white_level).contiguous()
    return raw, raw_pattern, raw_packed, black_level, white_level


def postprocess(raw, raw_pattern, im, bl, wl, output_bps=16):
    im = im * (wl - bl) + bl
    im = im.numpy()[0]
    im = depack_raw_bayer(im, raw_pattern)
    H, W = im.shape
    raw.raw_image_visible[:H, :W] = im
    rgb = raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=output_bps)
    rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    return rgb

@torch.no_grad()
def image_process():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--pretrained_network', type=str, required=True, help='the pretrained network path.')
    parser.add_argument('--data_path', type=str, required=True, help='the folder where contains only your raw images.')
    parser.add_argument('--save_path', type=str, default='inference/image_process', help='the folder where to save the processed images (in rgb), DEFAULT: \'inference/image_process\'')
    parser.add_argument('-opt', '--network_options', default='options/base/network_g/unet.yaml', help='the arch options of the pretrained network, DEFAULT: \'options/base/network_g/unet.yaml\'')
    parser.add_argument('--ratio', '--dgain', type=float, default=1.0, help='the ratio/additional digital gain you would like to add on the image, DEFAULT: 1.0.')
    parser.add_argument('--target_exposure', type=float, help='Target exposure, activate this will deactivate ratio.')
    parser.add_argument('--bps', '--output_bps', type=int, default=16, help='the bit depth for the output png file, DEFAULT: 16.')
    parser.add_argument('--led', action='store_true', help='if you are using a checkpoint fine-tuned by our led.')
    args = parser.parse_args()

    print('Building network...')
    network_g = build_network(yaml_load(args.network_options)['network_g'])
    print('Loading checkpoint...')
    load_network(network_g, args.pretrained_network, param_key='params' if not args.led else 'params_deploy')
    device = get_available_device()
    network_g = network_g.to(device)
    raw_paths = list(sorted(glob.glob(f'{args.data_path}/*')))
    ratio = args.ratio
    os.makedirs(args.save_path, exist_ok=True)

    for raw_path in tqdm(raw_paths):
        if args.target_exposure is not None:
            iso, exp_time = metainfo(raw_path)
            ratio = args.target_exposure / (iso * exp_time)
        raw, raw_pattern, im, bl, wl = read_img(raw_path)
        im = (im - bl) / (wl - bl)
        im = (im * ratio).clip(max=torch.tensor(1.0))
        im = im.to(device)

        result = network_g(im)

        result = result.clip(0, 1).cpu()
        rgb = postprocess(raw, raw_pattern, result, bl, wl, args.bps)
        cv2.imwrite(raw_path.replace(args.data_path, args.save_path)+'.png', rgb)
        raw.close()

if __name__ == '__main__':
    image_process()
