import argparse
import glob
import os
import time
from copy import deepcopy
import math

import cv2
import numpy as np
import rawpy
import torch
import torch.nn.functional as F
from tqdm import tqdm
import sys

from ultraled.archs import build_network
from ultraled.utils.options import yaml_load
from ultraled.data.raw_utils import *


def load_network(net, load_path, strict = True, param_key = 'params'):
    """Load network weights from checkpoint."""
    load_net = torch.load(load_path, map_location=lambda storage, loc: storage)
    
    if param_key is not None:
        if param_key not in load_net and 'params' in load_net:
            param_key = 'params'
            print('Loading: params_ema does not exist, use params.')
        load_net = load_net[param_key]
    
    print(f'Loading {net.__class__.__name__} model from {load_path}, with param key: [{param_key}].')
    
    for k, v in deepcopy(load_net).items():
        if k.startswith('module.'):
            load_net[k[7:]] = v
            load_net.pop(k)
    
    net.load_state_dict(load_net, strict=strict)


def get_available_device():
    """Get available computing device."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')



def setup_network(network_options, pretrained_path):
    """Setup and load network."""
    print('Building network...')
    print(network_options)
    network = build_network(yaml_load(network_options)['network_g'])
    
    print('Loading checkpoint...')
    load_network(network, pretrained_path)
    
    device = get_available_device()
    return network.to(device)


@torch.no_grad()
def image_process():
    """Main image processing pipeline."""
    parser = argparse.ArgumentParser(description='Image processing pipeline')
    parser.add_argument('-p', '--pretrained_network', type=str, required=True,
                       help='the pretrained ratio map estimator network path.')
    parser.add_argument('-pd', '--pretrained_denosing_network', type=str, required=True,
                       help='the pretrained network path for denoising.')
    parser.add_argument('--data_path', type=str, required=True,
                       help='the folder containing raw images to be processed.')
    parser.add_argument('--save_path', type=str, default='inference/image_process',
                       help='output folder for processed images.')
    parser.add_argument('-opt', '--network_options', 
                       default='options/base/network_g/cunet.yaml',
                       help='ratio map estimator network architecture options.')
    parser.add_argument('-optd', '--denoising_network_options',
                       default='options/base/network_g/41unet.yaml',
                       help='denoising network architecture options.')
    parser.add_argument('--ratio', '--dgain', type=float, default=1.0,
                       help='maximum exposure gain ratio.')
    parser.add_argument('--target_exposure', type=float,
                       help='target exposure (overrides ratio).')
    parser.add_argument('--bps', '--output_bps', type=int, default=8,
                       help='output bit depth.')
    
    args = parser.parse_args()

    device = get_available_device()

    network_g = setup_network(args.network_options, args.pretrained_network)
    network_gd = setup_network(args.denoising_network_options, args.pretrained_denosing_network)

    raw_paths = sorted(glob.glob(f'{args.data_path}/*'))
    ratio = args.ratio
    os.makedirs(args.save_path, exist_ok=True)

    for raw_path in tqdm(raw_paths, desc="Processing images"):
        start_time = time.time()
        
        if args.target_exposure is not None:
            iso, exp_time = metainfo(raw_path)
            ratio = args.target_exposure / (iso * exp_time)

        raw, raw_pattern, im, bl, wl = read_img(raw_path)
        im0 = (im - bl) / (wl - bl)
        im_normalized = im0 * ratio

        im_normalized = im_normalized.to(device)
        result = network_g(im_normalized)
        result = filter_bilateral(result, 15, torch.tensor(15.0).cuda(), torch.tensor(1.0).cuda())
        result = result.cpu()

        ratiomap_output = result
        realmap = torch.tensor(ratio / ratiomap_output).to(device)
        result = im0 * ratio / ratiomap_output
        
        result = result.clip(0.0, 1.0).to(device)

        ratio1 = realmap.mean().item()
        result1 = network_gd(result, realmap, if_train=False)
        result1 = result1.cpu().clip(0, 1)

        rgb = postprocess(raw, raw_pattern, result1, bl, wl, args.bps)
        rgb_tensor = torch.FloatTensor(rgb / 1.0)
        rgb_final = rgb_tensor.cpu().numpy().astype(np.uint8)
        
        base_save_path = raw_path.replace(args.data_path, args.save_path)
        cv2.imwrite(f'{base_save_path}.png', rgb_final)
        
        raw.close()


if __name__ == '__main__':
    image_process()