import numpy as np
import cv2
import os
import rawpy
import imageio
from sklearn.linear_model import LinearRegression

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


# def bayer2rggb(bayer):
#     H, W = bayer.shape
#     return bayer.reshape(H//2, 2, W//2, 2).transpose(0, 2, 1, 3).reshape(H//2, W//2, 4)

# read raw and return rgb
def raw2rgb(raw_path):
    raw_image = rawpy.imread(raw_path)
    rgb_image = raw_image.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=8)
    return rgb_image
    # imageio.imsave('output_rgb_image.jpg', rgb_image)



# def load_rgb_image(rgb_root):
#     rgb_path = os.path.join(rgb_root,"raw_visual")
#     filenames = os.listdir(rgb_path)
#     return cv2.imread(os.path.join(rgb_path,filenames[0]))


# Obtain and store the color block positions for a specific camera's color image
def get_block_positoins(cam_dir,iso=100):
    raw_path = os.path.join(os.path.join(cam_dir,'color'),str(iso))
    raw_imgs = sorted(os.listdir(raw_path))
    if '.DS_Store' in raw_imgs:
        raw_imgs.remove('.DS_Store')
    example_raw_img = os.path.join(raw_path,raw_imgs[0])
    color_example_rgb = raw2rgb(example_raw_img)
    block_positions = select_block_positions(color_example_rgb,calib_num=24)
    cam_name = cam_dir.split('/')[-1]
    save_file_name = f'{cam_name}_block_pos.npy'
    if not os.path.exists('./pos'):
        os.mkdir('./pos')
    np.save(os.path.join('./pos',save_file_name),block_positions)


# select each block and ret the positions
def select_block_positions(rgb_image,calib_num=24):
    positions = []
    for i in range(calib_num):
        # rect = cv2.selectROI(rgb_image,False,False)
        rect = cv2.selectROI(rgb_image,showCrosshair=True)
        cv2.rectangle(rgb_image,rect,color=(23,128,62))
        print(i)
        print(rect)
        print('---------------')
        positions.append(list(rect))

    return np.array(positions)

def select_whole_positions(rgb_image):
    rect = cv2.selectROI(rgb_image,showCrosshair=True)
    cv2.rectangle(rgb_image,rect,color=(23,128,62))

    return rect


def linear_regression(x, y): 
    return np.polyfit(x, y, 1)


def regr_plot(x, y, log=True, ax=None, c1=None, c2=None, label=False):
        x = np.array(x)
        y = np.array(y)
        if log:
            x = np.log(x)
            y = np.log(y)
        ax.scatter(x, y)

        regr = LinearRegression()
        regr.fit(x.reshape(-1,1), y)
        a, b = float(regr.coef_), float(regr.intercept_)   
        # ax.set_title('log(sigR) | log(K)')
        x_range = np.array([np.min(x), np.max(x)])
        std = np.mean((a*x+b-y)**2) ** 0.5
        
        if c1 is not None:
            if label:
                label = f'k={a:.5f}, b={b:.5f}, std={std:.5f}'
                ax.plot(x, regr.predict(x.reshape(-1,1)), linewidth = 2, c=c1, label=label)
            else:
                ax.plot(x, regr.predict(x.reshape(-1,1)), linewidth = 2, c=c1)
        
        if c2 is not None:
            ax.plot(x_range, a*x_range+b+std, c=c2, linewidth = 1)
            ax.plot(x_range, a*x_range+b-std, c=c2, linewidth = 1)

        data = {'k':a,'b':b,'sig':std}

        return ax, data


# raw2rgb('/Users/hyx/Code/CV/raw_denoising/calib/r10/color/100/IMG_0483.CR3')

# data = np.load('/Users/hyx/Code/CV/raw_denoising/pos/r10_block_pos.npy')
# print(data)

# block_positoins = np.load('/Users/hyx/Code/CV/raw_denoising/pos/r10_block_pos.npy')

# raw_path = os.path.join(os.path.join('/Users/hyx/Code/CV/raw_denoising/calib/r10','color'),str(100))
# raw_imgs = sorted(os.listdir(raw_path))
# raw_imgs.remove('.DS_Store')
# example_raw_img = os.path.join(raw_path,raw_imgs[0])
# color_example_rgb = raw2rgb(example_raw_img)

# for i in range(24):
#     # rect = cv2.selectROI(rgb_image,False,False)
#     rect = block_positoins[i]
#     cv2.rectangle(color_example_rgb,rect,color=(23,128,62))
# cv2.imshow('example',color_example_rgb)
# cv2.waitKey(0)

