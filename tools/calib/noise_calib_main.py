from utils import *
import rawpy
import os
import numpy as np
import matplotlib.pyplot as plt
import json
import scipy
from calib_color import *
from calib_dark import *

# fit (log) var_TL, var_gauss, var_row and K
def fit_log(cam_name):
    sig_gauss = []
    sig_row = []
    sig_TL = []
    K = []
    K_points = []
    with open(f'./result/calib/color_calib_params/{cam_name}/color_calib_params.json','r') as f:
        data = json.load(f)
        K_list, var_No_list = data['K_list'], data['var_No_list']
    with open(f'./result/calib/dark_calib_params/{cam_name}/dark_calib_params_channel_mean.json','r') as f:
        dark_calib_params_channel_mean = json.load(f)
    
    for iso, param in dark_calib_params_channel_mean.items():
            
            # for i in range(4):
        cur_k = np.mean(K_list[iso])
        K.append(cur_k)
        sig_row.append(np.mean(param['sigmaR']))
        sig_TL.append(np.mean(param['sigmaTL']))
        sig_gauss.append(np.mean(param['sigmaGs']))
            # K_points.append(K)
    
    fig = plt.figure(figsize=(20,8))

    axsig_row = fig.add_subplot(1,3,1)
    axsig_TL = fig.add_subplot(1,3,2)
    axsig_gauss = fig.add_subplot(1,3,3)

    axsig_row.set_title('log(sig_row) - log(K)')
    axsig_TL.set_title('log(sig_TL) - log(K)')
    axsig_gauss.set_title('log(sig_gauss) - log(K)')
    
    axsig_row, data_row = regr_plot(K, sig_row, ax=axsig_row, c1='red', c2='orange', log=True, label=True)
    axsig_TL, data_TL = regr_plot(K, sig_TL, ax=axsig_TL, c1='red', c2='orange', log=True, label=True)
    axsig_gauss, data_gauss = regr_plot(K, sig_gauss, ax=axsig_gauss, c1='red', c2='orange', log=True, label=True)

    # plt.show()
    params = dict()
    params['Kmin'] = min(min(values) for values in K_list.values())
    params['Kmax'] = max(max(values) for values in K_list.values())
    params['Row'] = dict()
    params['TL'] = dict()
    params['Gauss'] = dict()

    params['Row']['slope'] = data_row['k']
    params['Row']['bias'] = data_row['b']
    params['Row']['std'] = data_row['sig']

    params['TL']['slope'] = data_TL['k']
    params['TL']['bias'] = data_TL['b']
    params['TL']['std'] = data_TL['sig']

    params['Gauss']['slope'] = data_gauss['k']
    params['Gauss']['bias'] = data_gauss['b']
    params['Gauss']['std'] = data_gauss['sig']

    cam_param_dir = '.result/cam_log_params/' + cam_name + '.json'
    if not os.path.exists(cam_param_dir):
        os.mkdir(cam_param_dir)
    with open(cam_param_dir,'w') as json_file:
        json.dump(params,json_file)
    if not os.path.exists('.result/log_figs/'):
        os.mkdir('./result/log_figs/')
    plt.savefig('./result/log_figs/' + cam_name + '.png')


if __name__ == "__main__":
    # fit_log('6d2')
    # calib_color_per_camera(cam_dir='/Users/hyx/Code/CV/raw_denoising/calib/r10')
    # calib_dark()
    # get_block_positoins(cam_dir= '/Users/hyx/Code/CV/raw_denoising/calib/550d')
    # calib_color_per_iso_whole(cam_dir='/Users/hyx/Code/CV/raw_denoising/calib/6d2',iso=1600)
    # calib_dark_per_iso('/Users/hyx/Code/CV/raw_denoising/calib/d5200',1600)

    # calib_dark_per_camera('/Users/hyx/Code/CV/raw_denoising/calib/d5200')
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir',type=str,default='./calib')
    parser.add_argument('--mode',type=str)
    directory = parser.parse_args().dir
    cam_list = os.listdir(directory)
    if '.DS_Store' in cam_list:
        cam_list.remove('.DS_Store')
    print(cam_list)
    if parser.parse_args().mode == 'get_pos':
        for cam in cam_list:
            get_block_positoins(cam_dir=os.path.join(directory,cam))
    elif parser.parse_args().mode == 'calib':
        for cam in cam_list:
            cam_dir = os.path.join(directory,cam)
            calib_dark_per_camera(cam_dir)
            print('-------')
    elif parser.parse_args().mode == 'fig_log':
        for cam in cam_list:
            fit_log(cam)