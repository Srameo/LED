from glob import glob
import os
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from skimage.color import rgb2gray
from skimage import io
import numpy as np
import sys
import re



CAMERAS = ['Camera1', 'Camera2']


### for debug
def list_files(startpath, file=sys.stdout):
    for root, dirs, files in os.walk(startpath):
        level = root.replace(startpath, '').count(os.sep)
        indent = ' ' * 4 * (level)
        print('{}{}/'.format(indent, os.path.basename(root)))
        subindent = ' ' * 4 * (level + 1)
        for f in files:
            if 'tmp' in f or 'metadata' in f:
                continue
            print('{}{}'.format(subindent, f), file=file)


def calculate_ratio(lq, gt=None):
    shutter_lq = float(re.search(r'_(\d+(\.\d+)?)$', os.path.splitext(lq)[0]).group(0)[1:])
    shutter_gt = float(re.search(r'_(\d+(\.\d+)?)$', os.path.splitext(gt)[0]).group(0)[1:]) \
                 if gt is not None else 3000
    return shutter_gt / shutter_lq


def calculate_score(out, gt):
    def _score_base(_psnr, _ssim):
        return _psnr + np.log(_ssim) / np.log(1.2)

    # calculate metrics on rgb domain
    psnr_v = psnr(gt, out)
    ssim_v = ssim(rgb2gray(gt), rgb2gray(out))
    score = _score_base(psnr_v, ssim_v)

    return score, psnr_v, ssim_v


def main(input_dir, output_dir):
    input_folder = os.path.join(input_dir, 'res')
    if len(os.listdir(input_folder)) == 1:
        input_folder = os.path.join(input_folder, os.listdir(input_folder)[0])
    gt_folder = os.path.join(input_dir, 'ref')
    if len(os.listdir(gt_folder)) == 1:
        gt_folder = os.path.join(gt_folder, os.listdir(gt_folder)[0])

    output_filename = os.path.join(output_dir, 'scores.txt')

    total_status = [0, 0, 0]
    total_count = 0
    # lines = []
    for camera in CAMERAS:
        for path in sorted(glob(f'{input_folder}/{camera}/*.png')):
            # ratio = calculate_ratio(path)
            gt_im = io.imread(f'{gt_folder}/{camera}/long/{os.path.basename(path)}')
            out = io.imread(path)
            score_v, psnr_v, ssim_v = calculate_score(out, gt_im)
            total_status[0] += score_v
            total_status[1] += psnr_v
            total_status[2] += ssim_v
            # out_str = f'{camera}/{"/".join(os.path.basename(path).split("_")[:-1])} ratio: {ratio:.2f}:\n\t' \
                    #   f'PSNR: {psnr_v:.2f} SSIM: {ssim_v:.4f} Score: {score_v:.2f}\n'
            # lines.append(out_str)
            # print(out_str)
            total_count += 1

    with open(output_filename, 'w') as out_fp:
        out_fp.write('{}: {}\n'.format('PSNR',  total_status[1] / total_count))
        out_fp.write('{}: {}\n'.format('SSIM',  total_status[2] / total_count))
        out_fp.write('{}: {}\n'.format('Score', total_status[0] / total_count))
        out_fp.write('DEVICE: CPU')
        # out_fp.write('DETAILS:\n')
        # out_fp.writelines(lines)


if __name__ == '__main__':
    try:
        input_dir = sys.argv[1]
        output_dir = sys.argv[2]
        main(input_dir, output_dir)
    except Exception as e:
        print('Detailed files:', file=sys.stderr)
        list_files(input_dir, file=sys.stderr)
        print("", file=sys.stderr)
        raise e
