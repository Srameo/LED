from torch.utils import data as data
import numpy as np
import torch
import glob
from os import path as osp
from tqdm import tqdm

from led.utils.registry import DATASET_REGISTRY
from led.data.raw_utils import pack_raw_bayer

@DATASET_REGISTRY.register()
class RAWGTDataset(data.Dataset):
    def __init__(self, opt) -> None:
        super().__init__()
        self.opt = opt

        self.root_folder = opt['dataroot']
        self.postfix = opt.get('postfix', None)
        self.zero_clip = 0 if opt.get('zero_clip', True) else None
        self.ratio_range = list(opt.get('ratio_range', (100, 300)))
        self.ratio_range[-1] = self.ratio_range[-1] + 1
        ## previously only support npz
        # assert self.postfix == 'npz'
        self.data_paths = glob.glob(
            osp.join(self.root_folder, f'*.{self.postfix}' if self.postfix is not None else '*'))
        self.load_in_mem = opt.get('load_in_mem', False)
        if self.load_in_mem:
            self.datas = {
                data_path: self.depack_meta(data_path, self.postfix, True)
                for data_path in tqdm(self.data_paths, desc='loading data in mem...')
            }

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
            import rawpy
            ## using rawpy
            raw = rawpy.imread(meta_path)
            raw_vis = raw.raw_image_visible.copy()
            raw_pattern = raw.raw_pattern
            wb = np.array(raw.camera_whitebalance, dtype='float32').copy()
            wb /= wb[1]
            ccm = np.array(raw.rgb_camera_matrix[:3, :3],
                           dtype='float32').copy()
            black_level = np.array(raw.black_level_per_channel,
                                   dtype='float32').reshape(4, 1, 1)
            white_level = np.array(raw.camera_white_level_per_channel,
                                   dtype='float32').reshape(4, 1, 1)
            im = pack_raw_bayer(raw_vis, raw_pattern).astype('float32')
            raw.close()
        else:
            raise NotImplementedError

        if to_tensor:
            im = torch.from_numpy(im).float().contiguous()
            black_level = torch.from_numpy(black_level).float().contiguous()
            white_level = torch.from_numpy(white_level).float().contiguous()
            wb = torch.from_numpy(wb).float().contiguous()
            ccm = torch.from_numpy(ccm).float().contiguous()

        return im, \
               black_level, white_level, \
               wb, ccm

    def randint(self, *range):
        return torch.randint(*range, size=(1,)).item()

    def __getitem__(self, index):
        data_path = self.data_paths[index]
        ratio = self.randint(*self.ratio_range)

        if not self.load_in_mem:
            im, black_level, white_level, wb, ccm = self.depack_meta(data_path, self.postfix, True)
        else:
            im, black_level, white_level, wb, ccm = self.datas[data_path]

        if self.opt['crop_size'] is not None:
            _, H, W = im.shape
            crop_size = self.opt['crop_size']
            assert crop_size < H and crop_size < W
            if self.opt['phase'] == 'train':
                h_start = self.randint(0, H - crop_size)
                w_start = self.randint(0, W - crop_size)
            else:
                # center crop
                h_start = (H - crop_size) // 2
                w_start = (W - crop_size) // 2
            im_patch = im[:, h_start:h_start+crop_size, w_start:w_start+crop_size]
        else:
            im_patch = im

        lq_im_patch = im_patch
        gt_im_patch = im_patch

        return {
            'lq': lq_im_patch,
            'gt': gt_im_patch,
            'ratio': ratio,
            'black_level': black_level,
            'white_level': white_level,
            'wb': wb,
            'ccm': ccm,
            'lq_path': data_path,
            'gt_path': data_path,
        }

    def __len__(self):
        return len(self.data_paths)