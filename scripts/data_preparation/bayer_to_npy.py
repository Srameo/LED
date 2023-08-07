import numpy as np
import os
from multiprocessing import Pool
from os import path as osp
from tqdm import tqdm
import rawpy
from glob import glob
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', type=str, help='Path to the dataset.')
    parser.add_argument('--save-path', type=str, help='Path to save the dataset.')
    parser.add_argument('--suffix', type=str, help='The postfix of the data.')
    parser.add_argument('--n-thread', default=4, type=int, help='thread num for processing.')
    args = parser.parse_args()

    opt = {}
    opt['n_thread'] = args.n_thread

    # Sony long images
    opt['input_folder'] = args.data_path
    opt['save_folder'] = args.save_path
    opt['child_folder'] = ['long', 'short']
    suffix = args.suffix
    opt['suffix'] = suffix if suffix.startswith('.') else '.'+suffix
    bayer2npy(opt)

def bayer2npy(opt):
    """Crop images to subimages.

    Args:
        opt (dict): Configuration dict. It contains:
            input_folder (str): Path to the input folder.
            save_folder (str): Path to save folder.
            n_thread (int): Thread number.
    """
    for child in opt['child_folder']:
        input_folder = osp.join(opt['input_folder'], child)
        save_folder = osp.join(opt['save_folder'], child)
        suffix = opt['suffix']
        if not osp.exists(save_folder):
            os.makedirs(save_folder)
            print(f'mkdir {save_folder} ...')
        # else:
        #     print(f'Folder {save_folder} already exists. Exit.')
        #     sys.exit(1)

        img_list = glob(f'{input_folder}/*{suffix}')
        print(len(img_list))

        pbar = tqdm(total=len(img_list), unit='image', desc='Extract')
        pool = Pool(opt['n_thread'])
        for path in img_list:
            # worker(path, opt, save_folder)
            # pbar.update(1)
            pool.apply_async(worker, args=(path, opt, save_folder), callback=lambda arg: pbar.update(1))
        pool.close()
        pool.join()
        pbar.close()
    print('All processes done.')

def pack_raw_bayer(raw: np.ndarray, raw_pattern: np.ndarray):
    #pack Bayer image to 4 channels
    R = np.where(raw_pattern==0)
    G1 = np.where(raw_pattern==1)
    B = np.where(raw_pattern==2)
    G2 = np.where(raw_pattern==3)

    raw = raw.astype(np.uint16)
    out = np.stack((raw[R[0][0]::2,  R[1][0]::2], #RGBG
                    raw[G1[0][0]::2, G1[1][0]::2],
                    raw[B[0][0]::2,  B[1][0]::2],
                    raw[G2[0][0]::2, G2[1][0]::2]), axis=0).astype(np.uint16)

    return out

def readimg(path):
    raw = rawpy.imread(path)
    raw_vis = raw.raw_image_visible.copy()
    raw_pattern = raw.raw_pattern
    wb = np.array(raw.camera_whitebalance).copy()
    wb /= wb[1]
    cam2rgb = np.array(raw.rgb_camera_matrix[:3, :3]).copy()
    black_level = np.array(raw.black_level_per_channel).reshape(4, 1, 1)
    white_level = np.array(raw.camera_white_level_per_channel).reshape(4, 1, 1)
    raw.close()
    return pack_raw_bayer(raw_vis, raw_pattern), wb, cam2rgb, black_level, white_level

def cvt_metadata(metadata):
    return {
        'wb': metadata[0],
        'ccm': metadata[1],
        'black_level': metadata[2],
        'white_level': metadata[3],
    }

def worker(path, opt, save_folder):
    """Worker for each process.

    Args:
        path (str): Image path.
        opt (dict): Configuration dict. It contains:
            crop_size (int): Crop size.
            step (int): Step for overlapped sliding window.
            thresh_size (int): Threshold size. Patches whose size is lower
                than thresh_size will be dropped.
            save_folder (str): Path to save folder.
            compression_level (int): for cv2.IMWRITE_PNG_COMPRESSION.

    Returns:
        process_info (str): Process information displayed in progress bar.
    """
    # crop_size = opt['crop_size']
    # step = opt['step']
    # assert step % 2 == 0 and crop_size % 2 == 0
    # thresh_size = opt['thresh_size']
    img_name, extension = osp.splitext(osp.basename(path))

    img, *metadata = readimg(path)

    np.savez(
        osp.join(save_folder, f'{img_name}{extension}'),
        im=img,
        **cvt_metadata(metadata)
    )
    process_info = f'Processing {img_name} ...'
    return process_info


if __name__ == '__main__':
    main()
