import os
import glob
import numpy as np
import shutil
from tqdm import tqdm
import argparse
from led.data.raw_utils import metainfo

CAMERA_TO_SUFFIX = {
    # 'CanonEOS5D4': '.CR2',
    'CanonEOS70D': '.CR2',
    'CanonEOS700D': '.CR2',
    'NikonD850': '.nef',
    'SonyA7S2': '.ARW'
}


def transform_data_frame(camera_path, target_path):
    camera = os.path.basename(camera_path)
    if not target_path.endswith(camera):
        target_path = f'{target_path}/{camera}'
    if not os.path.exists(target_path):
        os.makedirs(f'{target_path}/short', exist_ok=True)
        os.makedirs(f'{target_path}/long', exist_ok=True)
    f = open(f'{target_path}/{camera}_list.txt', 'w')
    suffix = CAMERA_TO_SUFFIX[camera]
    files = glob.glob(f'{camera_path}/scene-*/*{suffix}')
    gt_ids = np.array([1, 6, 11, 16])
    for file in tqdm(files, desc=camera):
        scene, rawpath = file.split('/')[-2:]
        scene_id = int(scene.split('-')[-1])
        img_id = int(rawpath[:-len(suffix)].split('_')[-1])
        gt_id = gt_ids[np.argmin(np.abs(img_id - gt_ids))]
        if img_id == gt_id:
            continue
        gt_path = f'{camera_path}/{scene}/IMG_{gt_id:04d}{suffix}'
        gt_iso, expo = metainfo(gt_path)
        gt_expo = gt_iso * expo
        lq_iso, expo = metainfo(file)
        lq_expo = lq_iso * expo
        lq_tgt_path = f'{target_path}/short/{scene_id:05d}_{img_id:02d}_{lq_expo}s{suffix}'
        gt_tgt_path = f'{target_path}/long/{scene_id:05d}_{gt_id:02d}_{gt_expo}s{suffix}'
        shutil.copy2(gt_path, gt_tgt_path)
        shutil.copy2(file, lq_tgt_path)
        f.write(
            f'{os.path.relpath(lq_tgt_path, target_path)} {os.path.relpath(gt_tgt_path, target_path)} ISO{lq_iso} ISO{gt_iso}\n'
        )
        # print(scene_id, img_id, gt_id, gt_expo, lq_expo)
    f.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', type=str, help='Path to the ELD dataset.')
    parser.add_argument('--save-path', type=str, help='Path to save ELD dataset, which is in the SID data structure.')

    args = parser.parse_args()
    eld_path = args.data_path
    tgt_path = args.save_path
    for camera in CAMERA_TO_SUFFIX:
        transform_data_frame(f'{eld_path}/{camera}', tgt_path)
