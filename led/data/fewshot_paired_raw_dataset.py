import re
import rawpy
from torch.utils import data as data
import numpy as np
import torch
from os import path as osp
from tqdm import tqdm

from led.utils.registry import DATASET_REGISTRY
from led.data.raw_utils import pack_raw_bayer, Sony_A7S2_CCM, metainfo


@DATASET_REGISTRY.register()
class FewshotPairedRAWDataset(data.Dataset):
    def __init__(self, opt) -> None:
        super().__init__()
        self.opt = opt

        self.root_folder = opt['dataroot']
        self.which_meta = opt.get('which_meta', 'gt')
        self.zero_clip = 0 if opt.get('zero_clip', True) else None

        self.lq_paths, self.gt_paths, self.ratios = [], [], []
        with open(opt['data_pair_list'], 'r') as data_pair_list:
            pairs = data_pair_list.readlines()
            for pair in pairs:
                lq, gt = pair.split(' ')[:2]
                gt = gt.rstrip('\n')
                lq_path = osp.join(self.root_folder, lq)
                gt_path = osp.join(self.root_folder, gt)
                self.lq_paths.append(lq_path)
                self.gt_paths.append(gt_path)
                iso, exp_time = metainfo(lq_path)
                lq_ev = iso * exp_time
                iso, exp_time = metainfo(gt_path)
                gt_ev = iso * exp_time
                self.ratios.append(gt_ev / lq_ev)
        self.lqs = {
            data_path: \
                self.depack_meta(data_path)
            for data_path in tqdm(set(self.lq_paths), desc='load lq metas in mem...')
        }
        self.gts = {
            data_path: \
                self.depack_meta(data_path)
            for data_path in tqdm(set(self.gt_paths), desc='load gt metas in mem...')
        }


    @staticmethod
    def depack_meta(meta_path, to_tensor=True):
        ## using rawpy
        raw = rawpy.imread(meta_path)
        raw_vis = raw.raw_image_visible.copy()
        raw_pattern = raw.raw_pattern
        wb = np.array(raw.camera_whitebalance, dtype='float32').copy()
        wb /= wb[1]
        ccm = Sony_A7S2_CCM.copy()
        black_level = np.array(raw.black_level_per_channel,
                                dtype='float32').reshape(4, 1, 1)
        white_level = np.array(raw.camera_white_level_per_channel,
                                dtype='float32').reshape(4, 1, 1)
        im = pack_raw_bayer(raw_vis, raw_pattern).astype('float32')
        raw.close()

        if to_tensor:
            im = torch.from_numpy(im).float().contiguous()
            black_level = torch.from_numpy(black_level).float().contiguous()
            white_level = torch.from_numpy(white_level).float().contiguous()
            wb = torch.from_numpy(wb).float().contiguous()
            ccm = torch.from_numpy(ccm).float().contiguous()

        return (im - black_level) / (white_level - black_level), \
                wb, ccm

    def __getitem__(self, index):
        lq_path = self.lq_paths[index]
        gt_path = self.gt_paths[index]
        ratio = self.ratios[index]

        lq_im, lq_wb, lq_ccm = self.lqs[lq_path]
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

            if self.opt.get('ratio_aug') is not None:
                ratio_range = self.opt['ratio_aug']
                rand_ratio = torch.rand((1,)).item() * (ratio_range[1] - ratio_range[0]) + ratio_range[0]
                ## TODO: maybe there are some over-exposed?
                gt_im_patch = gt_im_patch / ratio * rand_ratio
                ratio = rand_ratio

        lq_im_patch = torch.clip(lq_im_patch * ratio, self.zero_clip, 1)
        gt_im_patch = torch.clip(gt_im_patch, 0, 1)

        return {
            'lq': lq_im_patch,
            'gt': gt_im_patch,
            'ratio': torch.tensor(ratio).float(),
            'wb': gt_wb if self.which_meta == 'gt' else lq_wb,
            'ccm': gt_ccm if self.which_meta == 'gt' else lq_ccm,
            'lq_path': lq_path,
            'gt_path': gt_path
        }

    def __len__(self):
        return len(self.lq_paths)