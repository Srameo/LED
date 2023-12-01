import numpy as np
import rawpy
from utils import *
import matplotlib.pyplot as plt
import json
import scipy

def calib_dark_per_iso(cam_dir,iso):
    cur_iso_path = os.path.join(os.path.join(cam_dir,'dark'), str(iso))
    raw_imgs = os.listdir(cur_iso_path)
    if '.DS_Store' in raw_imgs:
        raw_imgs.remove('.DS_Store')

    r = 400
    # Read
    sigma_read = np.zeros((4, len(raw_imgs)), dtype=np.float32)
    mean_read = np.zeros((4, len(raw_imgs)), dtype=np.float32)
    r2_read = np.zeros((4, len(raw_imgs)), dtype=np.float32)
    # row
    sigma_row = np.zeros(len(raw_imgs), dtype=np.float32)
    mean_row = np.zeros(len(raw_imgs), dtype=np.float32)
    r2_row = np.zeros(len(raw_imgs), dtype=np.float32)
    # TL
    sigma_TL = np.zeros((4, len(raw_imgs)), dtype=np.float32)
    mean_TL = np.zeros((4, len(raw_imgs)), dtype=np.float32)
    r2_TL = np.zeros((4, len(raw_imgs)), dtype=np.float32)
    lamda = np.zeros((4, len(raw_imgs)), dtype=np.float32)
    # Gauss
    sigma_gauss = np.zeros((4, len(raw_imgs)), dtype=np.float32)
    mean_gauss = np.zeros((4, len(raw_imgs)), dtype=np.float32)
    r2_gauss = np.zeros((4, len(raw_imgs)), dtype=np.float32)

    for i, raw_img in enumerate(raw_imgs):
        # print(i)
        cur_raw_path = os.path.join(cur_iso_path, raw_img)
        raw = rawpy.imread(cur_raw_path)
        black_level = raw.black_level_per_channel

        # print(f'bl:{black_level}')

        raw_vis = raw.raw_image_visible.copy()
        raw_pattern = raw.raw_pattern
        raw = raw.raw_image_visible.astype(np.float32)
        # rggb_img = bayer2_Rggb(raw)

        # Calculate R before converting to 4 channels
        raw -= np.mean(black_level)
        row_all = np.mean(raw[raw.shape[0]//2-r*2:raw.shape[0]//2+r*2,raw.shape[1]//2-r*2:raw.shape[1]//2+r*2], axis=1)
        _, (sig_row,u_row,r_row) = scipy.stats.probplot(row_all,rvalue=True)
        sigma_row[i] = sig_row
        mean_row[i] = u_row
        r2_row[i] = r_row**2

        # Convert the image into RGGB four channels
        rggb_img = pack_raw_bayer(raw_vis, raw_pattern)
        rggb_img = rggb_img.transpose(1,2,0).astype(np.int64)
        # Subtract the black level
        rggb_img -= black_level

        # Crop out a square with a side length of 800
        H, W = rggb_img.shape[:2]
        rggb_img = rggb_img[H//2-r:H//2+r, W//2-r:W//2+r, :]


        # Iterate over the 4 channels
        for c in range(4):
            cur_channel_img = rggb_img[:, :, c]

            # Calculate the variance of TL (or the variance of Gaussian) + the variance of row, here recorded as the variance of read
            _, (sig_read,u_read,r_read) = scipy.stats.probplot(cur_channel_img.reshape(-1),rvalue=True)
            sigma_read[c][i] = sig_read
            mean_read[c][i] = u_read
            r2_read[c][i] = r_read**2

            # Calculate TL
            row_all_cur_channel = np.mean(cur_channel_img,axis=1)
            cur_channel_img = cur_channel_img.astype(np.float32)
            cur_channel_img -= row_all_cur_channel.reshape(-1,1)
            X = cur_channel_img.reshape(-1)
            lam = scipy.stats.ppcc_max(X)
            lamda[c][i] = lam
            _, (sig_TL,u_TL,r_TL) = scipy.stats.probplot(X,dist=scipy.stats.tukeylambda(lam), rvalue=True)
            sigma_TL[c][i] = sig_TL
            mean_TL[c][i] = u_TL
            r2_TL[c][i] = r_TL**2

            # Calculate gauss
            _,(sig_gauss,u_gauss,r_gauss) = scipy.stats.probplot(X,rvalue=True)
            sigma_gauss[c][i] = sig_gauss
            mean_gauss[c][i] = u_gauss
            r2_gauss[c][i] = r_gauss**2
    param = {
            'black_level':black_level,
            'lam':lamda.tolist(),
            'sigmaRead':sigma_read.tolist(), 'meanRead':mean_read.tolist(), 'r2Gs':r2_read.tolist(),
            'sigmaR':sigma_row.tolist(), 'meanR':mean_row.tolist(), 'r2R':r2_row.tolist(),
            'sigmaTL':sigma_TL.tolist(), 'meanTL':mean_TL.tolist(), 'r2TL':r2_TL.tolist(),
            'sigmaGs':sigma_gauss.tolist(), 'meanGs':mean_gauss.tolist(), 'r2Gs':r2_gauss.tolist(),
        }
    
    param_channel_mean = {
            'black_level':black_level,
            'lam':np.mean(lamda,axis=0).tolist(),
            'sigmaRead':np.mean(sigma_read,axis=0).tolist(), 'meanRead':np.mean(mean_read,axis=0).tolist(), 'r2Gs':np.mean(r2_read,axis=0).tolist(),
            'sigmaR':sigma_row.tolist(), 'meanR':mean_row.tolist(), 'r2R':r2_row.tolist(),
            'sigmaTL':np.mean(sigma_TL,axis=0).tolist(), 'meanTL':np.mean(mean_TL,axis=0).tolist(), 'r2TL':np.mean(r2_TL,axis=0).tolist(),
            'sigmaGs':np.mean(sigma_gauss,axis=0).tolist(), 'meanGs':np.mean(mean_gauss,axis=0).tolist(), 'r2Gs':np.mean(r2_gauss,axis=0).tolist(),
        }

    return param, param_channel_mean


# get noise params from dark imgs per camera
def calib_dark_per_camera(cam_dir):
    dark_dir = os.path.join(cam_dir, 'dark')
    iso_list = sorted(os.listdir(dark_dir))
    if '.DS_Store' in iso_list:
        iso_list.remove('.DS_Store')
    # if '400' in iso_list:
    #     iso_list.remove('400')
    dark_calib_params = dict()
    dark_calib_params_channel_mean = dict()
    for iso in iso_list:
        print(iso)
        param, param_channel_mean = calib_dark_per_iso(cam_dir,iso=int(iso))
        dark_calib_params[iso] = param
        dark_calib_params_channel_mean[iso] = param_channel_mean
    
    
    cam_name = cam_dir.split('/')[-1]
    if not os.path.exists('./result/calib/dark_calib_params'):
        os.mkdir('./result/calib/dark_calib_params')
    dark_calib_params_dir = os.path.join('./result/calib/dark_calib_params',cam_name)
    if not os.path.exists(dark_calib_params_dir):
        os.mkdir(dark_calib_params_dir)

    with open(os.path.join(dark_calib_params_dir,'dark_calib_params.json'),'w') as json_file:
        json.dump(dark_calib_params,json_file)
    with open(os.path.join(dark_calib_params_dir,'dark_calib_params_channel_mean.json'),'w') as json_file:
        json.dump(dark_calib_params_channel_mean,json_file)