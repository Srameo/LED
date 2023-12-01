import numpy as np
import rawpy
from utils import *
import matplotlib.pyplot as plt
import json

# Fit K and Var(No) using the formula Var(D) = K(KI) + Var(No)
def calib_color_per_iso(cam_dir, iso):
    cur_iso_path = os.path.join(os.path.join(cam_dir,'color'), str(iso))
    raw_imgs = os.listdir(cur_iso_path)
    if '.DS_Store' in raw_imgs:
        raw_imgs.remove('.DS_Store')

    # Retrieve previously stored position information
    camera = cam_dir.split('/')[-1]
    block_position_path = f'./pos/{camera}_block_pos.npy'
    block_positoins = np.load(block_position_path)
    KI = np.zeros((4,len(block_positoins)))
    var_D = np.zeros((4,len(block_positoins)))
    for raw_img in raw_imgs:
        cur_raw_path = os.path.join(cur_iso_path, raw_img)
        raw = rawpy.imread(cur_raw_path)
        black_level = raw.black_level_per_channel
        white_point = raw.camera_white_level_per_channel

        raw_vis = raw.raw_image_visible.copy()
        raw_pattern = raw.raw_pattern
        raw = raw.raw_image_visible.astype(np.float32)
        # rggb_img = bayer2_Rggb(raw)
        rggb_img = pack_raw_bayer(raw_vis, raw_pattern)
        rggb_img = rggb_img.transpose(1,2,0).astype(np.int64)

        rggb_img -= black_level
        # rggb_img = rggb_img.astype(np.float32)
        # rggb_img /= (np.array(white_point) - np.array(black_level))
        
        for i, pos in enumerate(block_positoins):
            minx,miny,w,h = pos
            maxx,maxy = minx+w, miny+h
            minx //= 2
            miny //= 2
            maxx //= 2
            maxy //= 2
            KI[:,i] += np.mean(rggb_img[miny:maxy,minx:maxx,:],axis=(0,1))
            # plt.imshow(rggb_img[miny:maxy,minx:maxx,:].mean(-1))
            # print(rggb_img[miny:maxy,minx:maxx,:].max())
            # print(rggb_img[miny:maxy,minx:maxx,:].min())
            # plt.show()
            var_D[:,i] += np.var(rggb_img[miny:maxy,minx:maxx,:],axis=(0,1))
   
    KI /= len(raw_imgs)

    var_D /= len(raw_imgs)
    K, var_No = np.zeros((4)), np.zeros((4))
    for i in range(4):        
        K[i], var_No[i] = linear_regression(KI[i], var_D[i])
    print(iso)
    # plt.scatter(KI[1],var_D[1])
    # plt.show()
    for i in range(4):
        plt.scatter(KI[i],var_D[i])
        fig_dir = './figs/'
        if not os.path.exists(fig_dir):
            os.makedirs(fig_dir)
        fig_name = fig_dir + camera + '/' + f'{iso}_K{i}.png'
        plt.savefig(fig_name)
        plt.close()
    return K,var_No


def calib_color_per_iso_whole(cam_dir, iso):
    cur_iso_path = os.path.join(os.path.join(cam_dir,'color'), str(iso))
    raw_imgs = os.listdir(cur_iso_path)
    if '.DS_Store' in raw_imgs:
        raw_imgs.remove('.DS_Store')
    
    example_img_path = os.path.join(cur_iso_path,raw_imgs[0])
    rgb_img_example = raw2rgb(example_img_path)
    rect = select_whole_positions(rgb_img_example)
    # rect = (1514, 1018, 2543, 1897)
    minx,miny,w,h = rect
    maxx,maxy = minx+w, miny+h
    minx //= 2
    miny //= 2
    maxx //= 2
    maxy //= 2
    print(rect)
    total_imgs = np.zeros((len(raw_imgs),maxy-miny,maxx-minx,4))
    for i,raw_img in enumerate(raw_imgs):
        cur_raw_path = os.path.join(cur_iso_path, raw_img)
        raw = rawpy.imread(cur_raw_path)
        black_level = raw.black_level_per_channel
        white_point = raw.camera_white_level_per_channel

        raw_vis = raw.raw_image_visible.copy()
        raw_pattern = raw.raw_pattern
        raw = raw.raw_image_visible.astype(np.float32)
        # rggb_img = bayer2_Rggb(raw)
        rggb_img = pack_raw_bayer(raw_vis, raw_pattern)
        rggb_img = rggb_img.transpose(1,2,0).astype(np.int64)

        rggb_img -= black_level
        rggb_img = rggb_img.astype(np.float32)
        rggb_img /= (np.array(white_point) - np.array(black_level))
        
        total_imgs[i] = rggb_img[miny:maxy,minx:maxx,:]
    
    mean_img = np.mean(total_imgs,axis=0)
    var_img = np.var(total_imgs,axis=0)
    
    # K, var_No = np.zeros((4)), np.zeros((4))
    # for i in range(4):        
    #     K[i], var_No[i] = linear_regression(mean_img, var_D[i])
    plt.scatter(mean_img.reshape(-1,4)[:,1],var_img.reshape(-1,4)[:,1])
    plt.show()
    # return K,var_No

def calib_color_per_camera(cam_dir):
    color_dir = os.path.join(cam_dir, 'color')
    iso_list = sorted(os.listdir(color_dir))
    if '.DS_Store' in iso_list:
        iso_list.remove('.DS_Store')
    color_calib_params = dict()
    color_calib_params['K_list'], color_calib_params['var_No_list'] = dict(), dict()
    for iso in iso_list:
        K, var_No = calib_color_per_iso(cam_dir,iso=int(iso))
        color_calib_params['K_list'][iso] = K.tolist()
        color_calib_params['var_No_list'][iso] = var_No.tolist()
    
    cam_name = cam_dir.split('/')[-1]
    color_calib_params_dir = os.path.join('./result/calib/color_calib_params',cam_name)
    if not os.path.exists(color_calib_params_dir):
        os.makedirs(color_calib_params_dir)
    with open(os.path.join(color_calib_params_dir,'color_calib_params.json'),'w') as json_file:
        json.dump(color_calib_params,json_file)