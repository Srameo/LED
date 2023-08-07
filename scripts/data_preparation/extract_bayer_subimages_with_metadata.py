from multiprocessing import Pool
import numpy as np
import os
from os import path as osp
from tqdm import tqdm
import rawpy
from glob import glob
from led.utils.process import raw2rgb_v2
from torchvision.utils import save_image
import torch
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', type=str, help='Path to the dataset.')
    parser.add_argument('--save-path', type=str, help='Path to save the dataset.')
    parser.add_argument('--suffix', type=str, help='The postfix of the data.')
    parser.add_argument('--save-image', default='', type=str, help='save image for valid, \'\' denotes not saving.')

    parser.add_argument('--n-thread', default=4, type=int, help='thread num for processing.')
    parser.add_argument('--crop-size', default=512, type=int, help='the cropped image size.')
    parser.add_argument('--step', default=512, type=int, help='step while croping.')
    parser.add_argument('--thresh-size', default=512, type=int, help='the minimum size.')
    args = parser.parse_args()

    opt = {}
    opt['n_thread'] = args.n_thread
    opt['save_image'] = args.save_image
    opt['suffix'] = args.suffix

    opt['input_folder'] = f'{args.data_path}'
    opt['save_folder'] = f'{args.save_path}'
    opt['crop_size'] = args.crop_size
    opt['step'] = args.step
    opt['thresh_size'] = args.thresh_size
    extract_subimages(opt)


def extract_subimages(opt):
    """Crop images to subimages.

    Args:
        opt (dict): Configuration dict. It contains:
            input_folder (str): Path to the input folder.
            save_folder (str): Path to save folder.
            n_thread (int): Thread number.
    """
    input_folder = opt['input_folder']
    save_folder = opt['save_folder']
    suffix = opt['suffix']
    if not osp.exists(save_folder):
        os.makedirs(save_folder)
        print(f'mkdir {save_folder} ...')

    img_list = list(sorted(glob(f'{input_folder}/*{suffix}')))
    print(len(img_list))

    pbar = tqdm(total=len(img_list), unit='image', desc='Extract')
    pool = Pool(opt['n_thread'])
    for idx, path in enumerate(img_list):
        # if idx == 36:
        #     import ipdb
        #     ipdb.set_trace()
        pool.apply_async(worker, args=(path, opt), callback=lambda arg: pbar.update(1))
        # worker(path, opt)
        # pbar.update(1)
    pool.close()
    pool.join()
    pbar.close()
    print('All processes done.')


def pack_raw_bayer(raw: np.ndarray, raw_pattern: np.ndarray):
    #pack Bayer image to 4 channels
    if raw_pattern.size > 4 or raw_pattern.max() > 3:  # not bayer
        return None
    R = np.where(raw_pattern==0)
    G1 = np.where(raw_pattern==1)
    B = np.where(raw_pattern==2)
    G2 = np.where(raw_pattern==3)

    H, W = raw.shape
    if H % 2 == 1:
        H -= 1
    if W % 2 == 1:
        W -= 1

    out = np.stack((raw[R[0][0]:H:2,  R[1][0]:W:2], #RGBG
                    raw[G1[0][0]:H:2, G1[1][0]:W:2],
                    raw[B[0][0]:H:2,  B[1][0]:W:2],
                    raw[G2[0][0]:H:2, G2[1][0]:W:2]), axis=0).astype(np.uint16)

    return out

def readimg(path):
    raw = rawpy.imread(path)
    if raw.raw_type == rawpy.RawType.Stack: # not bayer
        return None, None
    raw_vis = raw.raw_image_visible.copy()
    raw_pattern = raw.raw_pattern
    wb = np.array(raw.camera_whitebalance).copy()
    wb /= wb[1]
    cam2rgb = np.array(raw.rgb_camera_matrix[:3, :3]).copy()
    black_level = np.array(raw.black_level_per_channel).reshape(4, 1, 1)
    white_level = np.array(raw.camera_white_level_per_channel)
    if (white_level == None).any():
        white_level = np.array(raw.white_level)
    if white_level.size == 1:
        white_level = white_level.repeat(4, 0)
    white_level = white_level.reshape(4, 1, 1)
    raw.close()
    return pack_raw_bayer(raw_vis, raw_pattern), wb, cam2rgb, black_level, white_level

def cvt_metadata(metadata):
    return {
        'wb': metadata[0],
        'ccm': metadata[1],
        'black_level': metadata[2],
        'white_level': metadata[3],
    }

def worker(path, opt):
    crop_size = opt['crop_size']
    step = opt['step']
    assert step % 2 == 0 and crop_size % 2 == 0
    thresh_size = opt['thresh_size']
    img_name, extension = osp.splitext(osp.basename(path))

    img, *metadata = readimg(path)
    if img is None:
        return f'Processing {img_name} failed!'

    if opt.get('save_image', '') != '':
        rgb = raw2rgb_v2(
            ((img.astype('float32') - metadata[2]) / (metadata[3] - metadata[2])).clip(0, 1),
            metadata[0], metadata[1])
        save_image(torch.from_numpy(rgb[None]), f'{opt["save_image"]}/{img_name}.jpg')

    h, w = img.shape[-2:]
    h_space = np.arange(0, h - crop_size + 1, step)
    if h - (h_space[-1] + crop_size) > thresh_size:
        h_space = np.append(h_space, h - crop_size)
    w_space = np.arange(0, w - crop_size + 1, step)
    if w - (w_space[-1] + crop_size) > thresh_size:
        w_space = np.append(w_space, w - crop_size)

    index = 0
    for x in h_space:
        for y in w_space:
            index += 1
            cropped_img = img[:, x:x + crop_size, y:y + crop_size]
            cropped_img = np.ascontiguousarray(cropped_img)
            np.savez(
                osp.join(opt['save_folder'], f'{img_name}_s{index:03d}{extension}'),
                im=cropped_img,
                **cvt_metadata(metadata)
            )
    process_info = f'Processing {img_name} ...'
    return process_info


if __name__ == '__main__':
    main()
