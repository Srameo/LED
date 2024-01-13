import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from skimage.color import rgb2gray


def calculate_score(out, gt):
    def _score_base(_psnr, _ssim):
        return _psnr + np.log(_ssim) / np.log(1.2)

    # calculate metrics on rgb domain
    psnr_v = psnr(gt, out)
    ssim_v = ssim(rgb2gray(gt), rgb2gray(out))
    score = _score_base(psnr_v, ssim_v)

    return score, psnr_v, ssim_v
