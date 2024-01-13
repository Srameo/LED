import os
import re
import torch
import numpy as np
from os import path as osp
from torch.utils import data as data
from tqdm import tqdm
from led.utils.registry import DATASET_REGISTRY

def calculate_ratio(lq, gt=None):
    """
    Calculate the additional dgain which needs to add on the input.
    The Exposure value is default set to 3000.
    How to calculate the exposure value can be found in `docs/demo.md`
    """
    shutter_lq = float(re.search(r'_(\d+(\.\d+)?)$', os.path.splitext(lq)[0]).group(0)[1:])
    shutter_gt = float(re.search(r'_(\d+(\.\d+)?)$', os.path.splitext(gt)[0]).group(0)[1:]) \
                 if gt is not None else 3000
    return shutter_gt / shutter_lq

def depack_meta(meta, to_tensor=True):
    """ Depack the npz file and normalize the input """

    ## load meta
    if isinstance(meta, str):
        meta = np.load(meta, allow_pickle=True)

    ## read meta data
    black_level = np.ascontiguousarray(meta['black_level'].copy().astype('float32'))
    white_level = np.ascontiguousarray(meta['white_level'].copy().astype('float32'))
    im = np.ascontiguousarray(meta['im'].copy().astype('float32'))
    wb = np.ascontiguousarray(meta['wb'].copy().astype('float32'))
    ccm = np.ascontiguousarray(meta['ccm'].copy().astype('float32'))
    meta.close()

    if to_tensor:
        ## convert to tensor
        im = torch.from_numpy(im).float().contiguous()
        black_level = torch.from_numpy(black_level).float().contiguous()
        white_level = torch.from_numpy(white_level).float().contiguous()
        wb = torch.from_numpy(wb).float().contiguous()
        ccm = torch.from_numpy(ccm).float().contiguous()

    return (im - black_level) / (white_level - black_level), wb, ccm


@DATASET_REGISTRY.register()
class MIPIDataset(data.Dataset):
    def __init__(self, opt) -> None:
        super().__init__()
        self.opt = opt

        self.root_folder = opt['dataroot']
        self.load_in_mem = opt.get('load_in_mem', False)

        ## load the data paths in this class
        self.lq_paths, self.gt_paths, self.ratios = [], [], []
        with open(opt['data_pair_list'], 'r') as data_pair_list:
            pairs = data_pair_list.readlines()
            for pair in pairs:
                lq, gt = pair.split(' ')[:2]
                gt = gt.rstrip('\n')
                ratio = calculate_ratio(lq, gt)
                self.lq_paths.append(osp.join(self.root_folder, lq))
                self.gt_paths.append(osp.join(self.root_folder, gt))
                self.ratios.append(ratio)

        if self.load_in_mem:
            # load data in mem
            self.lqs = {
                data_path: depack_meta(data_path)
                for data_path in tqdm(set(self.lq_paths), desc='load lq metas in mem...')
            }
            self.gts = {
                data_path: depack_meta(data_path)
                for data_path in tqdm(set(self.gt_paths), desc='load gt metas in mem...')
            }

    def __getitem__(self, index):
        lq_path = self.lq_paths[index]
        gt_path = self.gt_paths[index]
        ratio = self.ratios[index]

        if not self.load_in_mem:
            lq_im,     _,      _ = depack_meta(lq_path)
            gt_im, gt_wb, gt_ccm = depack_meta(gt_path)
        else:
            lq_im,     _,      _ = self.lqs[lq_path]
            gt_im, gt_wb, gt_ccm = self.gts[gt_path]

        ### augment
        ## crop
        if self.opt['crop_size'] is not None:
            _, H, W = lq_im.shape
            crop_size = self.opt['crop_size']
            assert crop_size <= H and crop_size <= W
            if self.opt['phase'] == 'train':
                h_start = torch.randint(0, H - crop_size, (1,)).item()
                w_start = torch.randint(0, W - crop_size, (1,)).item()
            else:
                # center crop
                h_start = (H - crop_size) // 2
                w_start = (W - crop_size) // 2
            lq_im_patch = lq_im[:, h_start:h_start+crop_size, w_start:w_start+crop_size]
            gt_im_patch = gt_im[:, h_start:h_start+crop_size, w_start:w_start+crop_size]
        else:
            lq_im_patch = lq_im
            gt_im_patch = gt_im
        ## flip + rotate
        if self.opt['phase'] == 'train':
            hflip = self.opt['use_hflip'] and torch.rand((1,)).item() < 0.5
            vflip = self.opt['use_rot'] and torch.rand((1,)).item() < 0.5
            rot90 = self.opt['use_rot'] and torch.rand((1,)).item() < 0.5
            if hflip:
                lq_im_patch = torch.flip(lq_im_patch, (2,))
                gt_im_patch = torch.flip(gt_im_patch, (2,))
            if vflip:
                lq_im_patch = torch.flip(lq_im_patch, (1,))
                gt_im_patch = torch.flip(gt_im_patch, (1,))
            if rot90:
                lq_im_patch = torch.permute(lq_im_patch, (0, 2, 1))
                gt_im_patch = torch.permute(gt_im_patch, (0, 2, 1))

        lq_im_patch = torch.clip(lq_im_patch * ratio, None, 1)
        gt_im_patch = torch.clip(gt_im_patch, 0, 1)

        return {
            'lq': lq_im_patch,
            'gt': gt_im_patch,
            'ratio': torch.tensor(ratio).float(),
            'wb': gt_wb,
            'ccm': gt_ccm,
            'lq_path': lq_path,
            'gt_path': gt_path
        }

    def __len__(self):
        return len(self.lq_paths)
