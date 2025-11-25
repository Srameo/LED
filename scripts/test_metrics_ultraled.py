import os
import glob
import numpy as np
import cv2
import torch
from torch import nn
import pyiqa
from tqdm import tqdm
import re
import sys


from led.data.raw_utils import *

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def match_image_pairs(folder_path_A, folder_path_B):
    pairs = []

    for fname in os.listdir(folder_path_A):
        if not fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            continue

        match = re.search(r'_(\d+)\.', fname)
        if (match and int(match.group(1)) == 10) or (match and int(match.group(1)) == 7):
            continue

        try:
            prefix = fname.split('_')[0]
            scene_num = int(prefix)
        except:
            continue

        target_fname = f"scene{scene_num}hdr.png"
        target_path = os.path.join(folder_path_B, target_fname)

        if os.path.exists(target_path):
            src_path = os.path.join(folder_path_A, fname)
            pairs.append((src_path, target_path))

    return pairs

def calculate_metrics_with_correction(folder_A, folder_B):
    psnr_metric = pyiqa.create_metric('psnr', device=device)
    ssim_metric = pyiqa.create_metric('ssim', device=device)
    lpips_metric = pyiqa.create_metric('lpips', device=device)
    
    image_pairs = match_image_pairs(folder_A, folder_B)

    if not image_pairs:
        print("No valid image pairs found")
        return

    total_psnr = 0.0
    total_ssim = 0.0
    total_lpips = 0.0
    valid_pairs = 0

    pbar = tqdm(image_pairs, desc="Processing")
    for a_path, b_path in pbar:
        try:
            img_a = load_image(a_path)
            img_b = load_image(b_path)
            
            img_a = img_a / 255.0
            img_b = img_b / 255.0

            if img_a.shape[:2] != img_b.shape[:2]:
                img_b = resize_image(img_b, target_shape=img_a.shape[:2], is_mask=False)

            tensor_a = image_to_tensor(img_a).clip(0, 1).to(device)
            tensor_b = image_to_tensor(img_b).clip(0, 1).to(device)

            corrected_a = illuminance_correct(tensor_a, tensor_b)
            corrected_a = corrected_a.clamp(0, 1)

            with torch.no_grad():
                psnr_score = psnr_metric(corrected_a, tensor_b)
                ssim_score = ssim_metric(corrected_a, tensor_b)
                lpips_score = lpips_metric(corrected_a, tensor_b)

            total_psnr += psnr_score.item()
            total_ssim += ssim_score.item()
            total_lpips += lpips_score.item()
            valid_pairs += 1

            current_metrics = {
                'PSNR': f"{psnr_score.item():.4f}",
                'SSIM': f"{ssim_score.item():.4f}", 
                'LPIPS': f"{lpips_score.item():.4f}"
            }

            pbar.set_postfix(current_metrics)

        except Exception as e:
            pbar.write(f"Failed {os.path.basename(a_path)}: {str(e)}")
            continue

    if valid_pairs > 0:
        avg_psnr = total_psnr / valid_pairs
        avg_ssim = total_ssim / valid_pairs
        avg_lpips = total_lpips / valid_pairs
        
        print(f"  PSNR:  {avg_psnr:.4f}", f"  SSIM:  {avg_ssim:.4f}", f"  LPIPS: {avg_lpips:.4f}")
        
        return {
            'psnr': avg_psnr,
            'ssim': avg_ssim,
            'lpips': avg_lpips,
            'valid_pairs': valid_pairs
        }
    else:
        print("No valid image pairs for evaluation")
        return None

if __name__ == "__main__":
    # Place test data in folder A and ground truth in folder B. When evaluating metrics, do not alter the naming format of source files.
    results = calculate_metrics_with_correction(
        folder_A="../../../defaultShare/archive/mengyuang/NeurIPS/final_test/ratio50_final_test",
        folder_B="../../../defaultShare/archive/mengyuang/SonyA7M4data_latest/groundtruth"
    )
    results = calculate_metrics_with_correction(
        folder_A="../../../defaultShare/archive/mengyuang/NeurIPS/final_test/ratio100_final_test",
        folder_B="../../../defaultShare/archive/mengyuang/SonyA7M4data_latest/groundtruth"
    )
    results = calculate_metrics_with_correction(
        folder_A="../../../defaultShare/archive/mengyuang/NeurIPS/final_test/ratio200_final_test",
        folder_B="../../../defaultShare/archive/mengyuang/SonyA7M4data_latest/groundtruth"
    )