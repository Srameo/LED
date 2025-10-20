import re
import math
from torch.utils import data as data
import numpy as np
import torch
from os import path as osp
from tqdm import tqdm
from ultraled.data.noise_util_rawhdr import NoiseGenerator



from ultraled.utils.registry import DATASET_REGISTRY


import torch
import math

import random
from ultraled.data.part_enhance import *


@DATASET_REGISTRY.register()
class EstimatorHDRRAWDataset(data.Dataset):
    def __init__(self, opt) -> None:
        super().__init__()
        self.opt = opt
        # print(self.opt)
        self.engine = self.opt.get('engine', 'torch')

        self.root_folder = opt['dataroot']
        self.postfix = opt.get('postfix', None)
        self.which_meta = opt.get('which_meta', 'gt')
        self.zero_clip = 0 if opt.get('zero_clip', True) else None
        self.ratio_range = opt.get('ratio_range', (100, 300))
        self.use_patches = opt.get('use_patches', False)
        self.load_in_mem = opt.get('load_in_mem', False)
        if self.use_patches:
            
            self.patch_id_max = opt.get('patch_id_max', 8)
            self.patch_tplt = opt.get('patch_tplt', '_s{:03}')
        
        assert self.postfix == 'npz'


        self.lq_paths, self.gt_paths = [], []
        with open(opt['data_pair_list'], 'r') as data_pair_list:
            pairs = data_pair_list.readlines()
            for pair in pairs:
                lq, gt = pair.split(' ')[:2]
                gt = gt.rstrip('\n')

                if not self.use_patches:
                    self.lq_paths.append(osp.join(self.root_folder, lq))
                    self.gt_paths.append(osp.join(self.root_folder, gt))

                else:
                    for i in range(1, 1 + self.patch_id_max):
                        self.lq_paths.append(osp.join(self.root_folder, self.insert_patch_id(lq, i, self.patch_tplt)))
                        self.gt_paths.append(osp.join(self.root_folder, self.insert_patch_id(gt, i, self.patch_tplt)))

        if self.load_in_mem:
            self.lqs = {
                '.'.join([data_path, self.postfix]): \
                    self.depack_meta('.'.join([data_path, self.postfix]), self.postfix)
                for data_path in tqdm(self.lq_paths, desc='load lq metas in mem...')
            }
            self.gts = {
                '.'.join([data_path, self.postfix]): \
                    self.depack_meta('.'.join([data_path, self.postfix]), self.postfix)
                for data_path in tqdm(self.gt_paths, desc='load gt metas in mem...')
            }

    @staticmethod
    def insert_patch_id(path, patch_id, tplt='_s{:03}'):
        exts = path.split('.')
        base = exts.pop(0)
        while exts[0] != 'ARW':
            base += '.' + exts.pop(0)
        base = base + tplt.format(patch_id)
        return base + '.' + '.'.join(exts)

    @staticmethod
    def depack_meta(meta_path, postfix='npz', to_tensor=True):
        if postfix == 'npz':
            meta = np.load(meta_path, allow_pickle=True)
            black_level = np.ascontiguousarray(meta['black_level'].copy().astype('float32'))
            white_level = np.ascontiguousarray(meta['white_level'].copy().astype('float32'))
            im = np.ascontiguousarray(meta['im'].copy().astype('float32'))
            wb = np.ascontiguousarray(meta['wb'].copy().astype('float32'))
            ccm = np.ascontiguousarray(meta['ccm'].copy().astype('float32'))
            meta.close()
        elif postfix == None:
            ## using rawpy
            raise NotImplementedError
        else:
            raise NotImplementedError

        if to_tensor:
            im = torch.from_numpy(im).float().contiguous()
            black_level = torch.from_numpy(black_level).float().contiguous()
            white_level = torch.from_numpy(white_level).float().contiguous()
            wb = torch.from_numpy(wb).float().contiguous()
            ccm = torch.from_numpy(ccm).float().contiguous()

        return (im - black_level) / (white_level - black_level), \
                wb, ccm

    @staticmethod
    def depack_meta_gt(meta_path, postfix='npz', to_tensor=True):
        if postfix == 'npz':
            meta = np.load(meta_path, allow_pickle=True)
            black_level = np.ascontiguousarray(meta['black_level'].copy().astype('float32'))
            white_level = np.ascontiguousarray(meta['white_level'].copy().astype('float32'))
            im = np.ascontiguousarray(meta['im'].copy().astype('float32'))
            wb = np.ascontiguousarray(meta['wb'].copy().astype('float32'))
            ccm = np.ascontiguousarray(meta['ccm'].copy().astype('float32'))
            meta.close()
        elif postfix == None:
            ## using rawpy
            raise NotImplementedError
        else:
            raise NotImplementedError

        if to_tensor:
            im = torch.from_numpy(im).float().contiguous()
            black_level = torch.from_numpy(black_level).float().contiguous()
            white_level = torch.from_numpy(white_level).float().contiguous()
            wb = torch.from_numpy(wb).float().contiguous()
            ccm = torch.from_numpy(ccm).float().contiguous()

        return im, \
                wb, ccm


    def randint(self, *range):
        if self.engine == 'torch':
            return torch.randint(*range, size=(1,)).item()
        else:
            return np.random.randint(*range)

    def flip(self, x, dim):
        if self.engine == 'torch':
            return torch.flip(x, (dim,))
        else:
            return np.flip(x, dim)

    def transpose(self, x):
        if self.engine == 'torch':
            return torch.permute(x, (0, 2, 1))
        else:
            return np.transpose(x, (0, 2, 1))

    def __getitem__(self, index):

        def sum_img_and_noise(img, noises):
            for noise in noises:
                img += noise
            return img
        

        lq_path = self.lq_paths[index]
        gt_path = self.gt_paths[index]
        ratio = self.randint(*self.ratio_range)

        if self.postfix is not None:
            lq_path = '.'.join([lq_path, self.postfix])
            gt_path = '.'.join([gt_path, self.postfix])


        if not self.load_in_mem:
            lq_im, lq_wb, lq_ccm = self.depack_meta(lq_path, self.postfix)
            gt_im, gt_wb, gt_ccm = self.depack_meta_gt(gt_path, self.postfix)
        else:
            lq_im, lq_wb, lq_ccm = self.lqs[lq_path]
            gt_im, gt_wb, gt_ccm = self.gts[gt_path]

        
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
                
                gt_im_patch = gt_im_patch / ratio * rand_ratio
                ratio = rand_ratio



        im_ratio = gt_im_patch.mean() // lq_im_patch.mean()
        lq_im_patch = gt_im_patch / im_ratio
        ratio_all = im_ratio * ratio
        _, H, W = gt_im_patch.shape

        Highlight_Generator = EstimatorHighlightGenerator(ratio_all)
        mask, addmap, lq_im_patch1 = Highlight_Generator.generate_highlight(lq_im_patch)
        mask = mask.unsqueeze(0)

        lq_im_patch = lq_im_patch.to(gt_im_patch.device)
        lq_im_patch = lq_im_patch * addmap

        map_ratio =  (lq_im_patch * ratio_all ) / (gt_im_patch * ratio + 1e-8)
        ratiomap_output = torch.mean(map_ratio, dim=0, keepdim=True).unsqueeze(0) * addmap
        
        lq_ev0_non_zero = lq_im_patch * im_ratio
        exposures = [
            torch.clip(lq_ev0_non_zero, min=0, max=1),
            torch.clip(lq_ev0_non_zero / 200, min=0, max=1),
            torch.clip(lq_ev0_non_zero / 120, min=0, max=1),
            torch.clip(lq_ev0_non_zero / 60, min=0, max=1),
            torch.clip(lq_ev0_non_zero / 32, min=0, max=1),
            torch.clip(lq_ev0_non_zero / 16, min=0, max=1),
            torch.clip(lq_ev0_non_zero / 8, min=0, max=1),
            torch.clip(lq_ev0_non_zero / 4, min=0, max=1),
            torch.clip(lq_ev0_non_zero / 2, min=0, max=1),
        ]

        exposures_gt = torch.stack(exposures)
        lq_nonoise = lq_im_patch * im_ratio
        gt_im_patch = torch.clip(gt_im_patch, min=0)
        gt_im_patch = ratiomap_output.squeeze(0)






        return {
            'lq_clean':lq_nonoise,
            'gt': exposures_gt,
            'ratio': torch.tensor(ratio),
            'ratio1': ratio_all,
            'wb': gt_wb if self.which_meta == 'gt' else lq_wb,
            'ccm': gt_ccm if self.which_meta == 'gt' else lq_ccm,
            'lq_path': lq_path,
            'gt_path': gt_path,
            'intact': True
        }

    def __len__(self):
        return len(self.lq_paths)
    


@DATASET_REGISTRY.register()
class DenoiserHDRRAWDataset(data.Dataset):
    def __init__(self, opt) -> None:
        super().__init__()
        self.opt = opt
        self.engine = self.opt.get('engine', 'torch')

        self.root_folder = opt['dataroot']
        self.postfix = opt.get('postfix', None)
        self.which_meta = opt.get('which_meta', 'gt')
        self.zero_clip = 0 if opt.get('zero_clip', True) else None
        self.ratio_range = opt.get('ratio_range', (100, 300))
        self.use_patches = opt.get('use_patches', False)
        self.load_in_mem = opt.get('load_in_mem', False)
        if self.use_patches:
            self.patch_id_max = opt.get('patch_id_max', 8)
            self.patch_tplt = opt.get('patch_tplt', '_s{:03}')

        assert self.postfix == 'npz'


        self.lq_paths, self.gt_paths = [], []
        with open(opt['data_pair_list'], 'r') as data_pair_list:
            pairs = data_pair_list.readlines()
            for pair in pairs:
                lq, gt = pair.split(' ')[:2]
                gt = gt.rstrip('\n')

                if not self.use_patches:
                    self.lq_paths.append(osp.join(self.root_folder, lq))
                    self.gt_paths.append(osp.join(self.root_folder, gt))

                else:
                    for i in range(1, 1 + self.patch_id_max):
                        self.lq_paths.append(osp.join(self.root_folder, self.insert_patch_id(lq, i, self.patch_tplt)))
                        self.gt_paths.append(osp.join(self.root_folder, self.insert_patch_id(gt, i, self.patch_tplt)))

        if self.load_in_mem:
            self.lqs = {
                '.'.join([data_path, self.postfix]): \
                    self.depack_meta('.'.join([data_path, self.postfix]), self.postfix)
                for data_path in tqdm(self.lq_paths, desc='load lq metas in mem...')
            }
            self.gts = {
                '.'.join([data_path, self.postfix]): \
                    self.depack_meta('.'.join([data_path, self.postfix]), self.postfix)
                for data_path in tqdm(self.gt_paths, desc='load gt metas in mem...')
            }

    @staticmethod
    def insert_patch_id(path, patch_id, tplt='_s{:03}'):
        exts = path.split('.')
        base = exts.pop(0)
        while exts[0] != 'ARW':
            base += '.' + exts.pop(0)
        base = base + tplt.format(patch_id)
        return base + '.' + '.'.join(exts)

    @staticmethod
    def depack_meta(meta_path, postfix='npz', to_tensor=True):
        if postfix == 'npz':
            meta = np.load(meta_path, allow_pickle=True)
            black_level = np.ascontiguousarray(meta['black_level'].copy().astype('float32'))
            white_level = np.ascontiguousarray(meta['white_level'].copy().astype('float32'))
            im = np.ascontiguousarray(meta['im'].copy().astype('float32'))
            wb = np.ascontiguousarray(meta['wb'].copy().astype('float32'))
            ccm = np.ascontiguousarray(meta['ccm'].copy().astype('float32'))
            meta.close()
        elif postfix == None:
            raise NotImplementedError
        else:
            raise NotImplementedError

        if to_tensor:
            im = torch.from_numpy(im).float().contiguous()
            black_level = torch.from_numpy(black_level).float().contiguous()
            white_level = torch.from_numpy(white_level).float().contiguous()
            wb = torch.from_numpy(wb).float().contiguous()
            ccm = torch.from_numpy(ccm).float().contiguous()

        return (im - black_level) / (white_level - black_level), \
                wb, ccm

    @staticmethod
    def depack_meta_gt(meta_path, postfix='npz', to_tensor=True):
        if postfix == 'npz':
            meta = np.load(meta_path, allow_pickle=True)
            black_level = np.ascontiguousarray(meta['black_level'].copy().astype('float32'))
            white_level = np.ascontiguousarray(meta['white_level'].copy().astype('float32'))
            im = np.ascontiguousarray(meta['im'].copy().astype('float32'))
            wb = np.ascontiguousarray(meta['wb'].copy().astype('float32'))
            ccm = np.ascontiguousarray(meta['ccm'].copy().astype('float32'))
            meta.close()
        elif postfix == None:
            ## using rawpy
            raise NotImplementedError
        else:
            raise NotImplementedError

        if to_tensor:
            im = torch.from_numpy(im).float().contiguous()
            black_level = torch.from_numpy(black_level).float().contiguous()
            white_level = torch.from_numpy(white_level).float().contiguous()
            wb = torch.from_numpy(wb).float().contiguous()
            ccm = torch.from_numpy(ccm).float().contiguous()

        return im, \
                wb, ccm


    def randint(self, *range):
        if self.engine == 'torch':
            return torch.randint(*range, size=(1,)).item()
        else:
            return np.random.randint(*range)

    def flip(self, x, dim):
        if self.engine == 'torch':
            return torch.flip(x, (dim,))
        else:
            return np.flip(x, dim)

    def transpose(self, x):
        if self.engine == 'torch':
            return torch.permute(x, (0, 2, 1))
        else:
            return np.transpose(x, (0, 2, 1))

    def __getitem__(self, index):

        def sum_img_and_noise(img, noises):
            for noise in noises:
                img += noise
            return img
        

        lq_path = self.lq_paths[index]
        gt_path = self.gt_paths[index]
        ratio = self.randint(*self.ratio_range)

        if self.postfix is not None:
            lq_path = '.'.join([lq_path, self.postfix])
            gt_path = '.'.join([gt_path, self.postfix])

        if not self.load_in_mem:
            lq_im, lq_wb, lq_ccm = self.depack_meta(lq_path, self.postfix)
            gt_im, gt_wb, gt_ccm = self.depack_meta_gt(gt_path, self.postfix)
        else:
            lq_im, lq_wb, lq_ccm = self.lqs[lq_path]
            gt_im, gt_wb, gt_ccm = self.gts[gt_path]

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
                
                gt_im_patch = gt_im_patch / ratio * rand_ratio
                ratio = rand_ratio



        im_ratio = gt_im_patch.mean() // lq_im_patch.mean()
        lq_im_patch = gt_im_patch / im_ratio
        ratio_all = im_ratio * ratio
        _, H, W = gt_im_patch.shape


        Highlight_Generator = DenoiserHighlightGenerator(ratio_all)
        mask, addmap, lq_im_patch1 = Highlight_Generator.generate_highlight(lq_im_patch)
        mask = mask.unsqueeze(0)

        lq_im_patch = lq_im_patch.to(gt_im_patch.device)
        lq_im_patch = lq_im_patch * addmap

        map_ratio =  (lq_im_patch * ratio_all ) / (gt_im_patch * ratio + 1e-8)
        ratiomap_output = torch.mean(map_ratio, dim=0, keepdim=True).unsqueeze(0) * addmap


        
        lq_ev0_non_zero = lq_im_patch * im_ratio
        exposures = [
            torch.clip(lq_ev0_non_zero, min=0, max=1),
            torch.clip(lq_ev0_non_zero / 200, min=0, max=1),
            torch.clip(lq_ev0_non_zero / 120, min=0, max=1),
            torch.clip(lq_ev0_non_zero / 60, min=0, max=1),
            torch.clip(lq_ev0_non_zero / 32, min=0, max=1),
            torch.clip(lq_ev0_non_zero / 16, min=0, max=1),
            torch.clip(lq_ev0_non_zero / 8, min=0, max=1),
            torch.clip(lq_ev0_non_zero / 4, min=0, max=1),
            torch.clip(lq_ev0_non_zero / 2, min=0, max=1),
        ]

        exposures_gt = torch.stack(exposures)
        lq_nonoise = lq_im_patch * im_ratio
        gt_im_patch = torch.clip(gt_im_patch, min=0)
        gt_im_patch = ratiomap_output.squeeze(0)






        return {
            'lq_clean':lq_nonoise,
            'gt': exposures_gt,
            'ratio': torch.tensor(ratio),
            'ratio1': ratio_all,
            'wb': gt_wb if self.which_meta == 'gt' else lq_wb,
            'ccm': gt_ccm if self.which_meta == 'gt' else lq_ccm,
            'lq_path': lq_path,
            'gt_path': gt_path,
            'intact': True
        }

    def __len__(self):
        return len(self.lq_paths)