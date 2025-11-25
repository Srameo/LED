import os
import math
import cv2
import rawpy
import torch
import exifread
import exifread
import numpy as np
from torch import nn
from copy import deepcopy
import torch.nn.functional as F


def read_img(raw_path):
    """Read and process raw image."""
    raw = rawpy.imread(raw_path)
    raw_vis = raw.raw_image_visible.copy()
    raw_pattern = raw.raw_pattern
    
    # Process black and white levels
    black_level = np.array(raw.black_level_per_channel, dtype=np.float32).reshape(1, 4, 1, 1)
    white_level = np.array(raw.camera_white_level_per_channel, dtype=np.float32)
    
    if (white_level == None).any():
        white_level = np.array(raw.white_level, dtype=np.float32)
    if white_level.size == 1:
        white_level = white_level.repeat(4, 0)
        
    white_level = white_level.reshape(1, 4, 1, 1)
    raw_packed = torch.from_numpy(np.float32(pack_raw_bayer(raw_vis, raw_pattern))[np.newaxis]).contiguous()
    black_level = torch.from_numpy(black_level).contiguous()
    white_level = torch.from_numpy(white_level).contiguous()
    
    return raw, raw_pattern, raw_packed, black_level, white_level


def postprocess(raw, raw_pattern, im, bl, wl, output_bps = 8):
    """Post-process the image to RGB."""
    im = im * (wl - bl) + bl
    im = im.numpy()[0]
    im = depack_raw_bayer(im, raw_pattern)
    
    H, W = im.shape
    raw.raw_image_visible[:H, :W] = im
    rgb = raw.postprocess(use_camera_wb=True, half_size=False, 
                         no_auto_bright=True, output_bps=output_bps)
    rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    
    return rgb


def filter_bilateral(tenIn, intSize, tenSigmas, tenSigmac):
    """Bilateral filter implementation."""
    tenSigmas = tenSigmas.view(-1, 1, 1, 1, 1)
    tenSigmac = tenSigmac.view(-1, 1, 1, 1, 1)

    # Create coordinate grids
    half_size = int(math.floor(0.5 * intSize))
    coords = torch.linspace(-half_size, half_size, intSize, 
                           dtype=tenIn.dtype, device=tenIn.device)
    tenHor = coords.view(1, -1).repeat(intSize, 1)
    tenVer = coords.view(-1, 1).repeat(1, intSize)

    # Calculate distances
    tenDists = (tenHor.square() + tenVer.square()).sqrt().view(1, 1, intSize * intSize, 1, 1)
    tenDistc = tenIn.view(tenIn.shape[0], tenIn.shape[1], 1, tenIn.shape[2], tenIn.shape[3])

    # Apply bilateral filtering
    tenOut = F.pad(input=tenIn, pad=[half_size, half_size, half_size, half_size], mode='reflect')
    tenOut = F.unfold(input=tenOut, kernel_size=intSize, stride=1, padding=0)
    tenOut = tenOut.view(tenIn.shape[0], tenIn.shape[1], intSize * intSize, tenIn.shape[2], tenIn.shape[3])
    
    tenWeight = ((-0.5 * tenDists.square() / (tenSigmas.square() + 1e-8)) + 
                (-0.5 * (tenOut - tenDistc).mean([1], True).square() / (tenSigmac.square() + 1e-8))).exp()
    tenWeight = tenWeight / (tenWeight.sum([2], True) + 1e-8)
    tenOut = (tenOut * tenWeight).sum([2], False)

    return tenOut



Sony_A7S2_CCM = np.array([[ 1.9712269,-0.6789218, -0.29230508],
                          [-0.29104823, 1.748401 , -0.45735288],
                          [ 0.02051281,-0.5380369,  1.5175241 ]],
                         dtype='float32')


Sony_A7S2_CCM = np.array([[ 1.9712269,-0.6789218, -0.29230508],
                          [-0.29104823, 1.748401 , -0.45735288],
                          [ 0.02051281,-0.5380369,  1.5175241 ]],
                         dtype='float32')


def pack_raw_bayer(raw: np.ndarray, raw_pattern: np.ndarray):
    #pack Bayer image to 4 channels
    R = np.where(raw_pattern==0)
    G1 = np.where(raw_pattern==1)
    B = np.where(raw_pattern==2)
    G2 = np.where(raw_pattern==3)

    raw = raw.astype(np.uint16)
    H, W = raw.shape
    if H % 2 == 1:
        raw = raw[:-1]
    if W % 2 == 1:
        raw = raw[:, :-1]
    out = np.stack((raw[R[0][0]::2,  R[1][0]::2], #RGBG
                    raw[G1[0][0]::2, G1[1][0]::2],
                    raw[B[0][0]::2,  B[1][0]::2],
                    raw[G2[0][0]::2, G2[1][0]::2]), axis=0).astype(np.uint16)

    return out


def depack_raw_bayer(raw: np.ndarray, raw_pattern: np.ndarray):
    _, H, W = raw.shape
    raw = raw.astype(np.uint16)

    R = np.where(raw_pattern==0)
    G1 = np.where(raw_pattern==1)
    B = np.where(raw_pattern==2)
    G2 = np.where(raw_pattern==3)

    raw_flatten = np.zeros((H * 2, W * 2))
    raw_flatten[R[0][0]::2,  R[1][0]::2] = raw[0]
    raw_flatten[G1[0][0]::2,  G1[1][0]::2] = raw[1]
    raw_flatten[B[0][0]::2,  B[1][0]::2] = raw[2]
    raw_flatten[G2[0][0]::2,  G2[1][0]::2] = raw[3]

    raw_flatten = raw_flatten.astype(np.uint16)
    return raw_flatten


def metainfo(rawpath):
    with open(rawpath, 'rb') as f:
        tags = exifread.process_file(f)
        _, suffix = os.path.splitext(os.path.basename(rawpath))

        if suffix == '.dng':
            expo = eval(str(tags['Image ExposureTime']))
            iso = eval(str(tags['Image ISOSpeedRatings']))
        else:
            expo = eval(str(tags['EXIF ExposureTime']))
            iso = eval(str(tags['EXIF ISOSpeedRatings']))

        # print('ISO: {}, ExposureTime: {}'.format(iso, expo))
    return iso, expo


def illuminance_correct(x, y):
    x_m = x.mean(dim=(-1, -2))
    y_m = y.mean(dim=(-1, -2))
    xy_m = (x * y).mean(dim=(-1, -2))
    xx_m = (x * x).mean(dim=(-1, -2))
    a = (xy_m - x_m * y_m) / (xx_m - x_m * x_m)
    b = y_m - a * x_m
    return a.reshape(1, -1, 1, 1) * x + b.reshape(1, -1, 1, 1)

def resize_image(img, target_shape, is_mask=False):
    h, w = target_shape
    interpolation = cv2.INTER_NEAREST if is_mask else cv2.INTER_AREA
    return cv2.resize(img, (w, h), interpolation=interpolation)

def load_image(path, target_size=None):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Cannot read image {path}")

    if len(img.shape) == 2:
        img = np.expand_dims(img, axis=-1)

    if img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if target_size is not None:
        img = cv2.resize(img, (target_size[1], target_size[0]), interpolation=cv2.INTER_AREA)

    return img.astype(np.float32)

def image_to_tensor(img):
    return torch.from_numpy(img).permute(2,0,1).unsqueeze(0)