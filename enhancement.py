import os
import cv2
import numpy as np
from tqdm import trange
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity
import argparse
import json

LOW_LIGHT_PATH = os.path.join('./images/low')
HIGH_LIGHT_PATH = os.path.join('./images/high')
JSON_PATH = os.path.join('./')

def get_bounds(x, Max, distance=3):
    """
    returns the lower and upper bound of the given position with distance
    """
    x_min = max(x - distance, 0)
    x_max = min(x + distance + 1, Max)
    return x_min, x_max

def esti_dark_channel(img, H, W, D=3):
    dark_channel = np.zeros((H, W))
    for i in trange(H):
        for j in range(W):
            top, bottom = get_bounds(i, H, D)
            left, right = get_bounds(j, W, D)
            local = img[top:bottom, left:right, :]
            dark_channel[i, j] = min(np.min(local, axis=(0, 1)))
    return dark_channel

def esti_atmos_light(img, dark_channel, H, W):
    total_pixels = H * W
    count = int(total_pixels * 0.001)
    index_tuple = np.unravel_index(np.argsort(-dark_channel, axis=None)[:count], dark_channel.shape)
    max_light_index = np.argmax(np.sum(img[index_tuple], axis=-1))
    atmospheric_light = img[index_tuple[0][max_light_index], index_tuple[1][max_light_index]]
    return atmospheric_light
    
def esti_trans(img, atmospheric_light, H, W, OMEGA=0.95, D=3):
    tmp = img.astype(np.float64) / atmospheric_light
    transmission = np.zeros((H, W))
    for i in trange(H):
        for j in range(W):
            top, bottom = get_bounds(i, H, D)
            left, right = get_bounds(j, W, D)
            local = tmp[top:bottom, left:right, :]
            transmission[i, j] =  1 - OMEGA * min(np.min(local, axis=(0, 1)))
    return transmission

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--omega', type=int, default=95, help='omega')
    parser.add_argument('-d', '--distance', type=int, default=3, help='distance')
    parser.add_argument('-t', '--threshold', type=int, default=10, help='threshold')
    parser.add_argument('-c', action='store_true', help='change the default parameters')
    return parser.parse_args()

if __name__ == '__main__':
    D = 3 # distance
    T0 = 0.1 # threshold
    OMEGA = 0.95
    args = parse_arguments()
    if args.c: 
        D = args.distance
        T0 = args.threshold / 100
        OMEGA = args.omega / 100
        img = cv2.imread(LOW_LIGHT_PATH + '/1.png')
        H, W, _ = img.shape # (400, 600, 3)
        rev_img = 255 - img
        dark_channel = esti_dark_channel(rev_img, H, W, D)
        atmospheric_light = esti_atmos_light(rev_img, dark_channel, H, W)
        transmission = esti_trans(rev_img, atmospheric_light, H, W, OMEGA, D)
        img_no_frog = (rev_img.astype(np.int32) - atmospheric_light.astype(np.int32)) \
                                    / np.expand_dims(np.clip(transmission, T0, 1.0), -1) + atmospheric_light
        img_no_frog = np.clip(img_no_frog, 0, 255).astype(np.uint8)
        img_enhance = 255 - img_no_frog
        cv2.imwrite('images/1_enhanced_' + str(D) + '_' + str(T0) + '_' + str(OMEGA) + '.png', img_enhance)
        imgHigh = cv2.imread(HIGH_LIGHT_PATH + '/1.png')
        psnr = peak_signal_noise_ratio(imgHigh, img_enhance)
        ssim = structural_similarity(imgHigh, img_enhance, channel_axis=2)
        test_dict = {'distance': D, 'threshold': T0, 'omega': OMEGA, 'PSNR': psnr, 'SSIM': ssim}
        with open(JSON_PATH + 'test.json', 'a') as f:
            json.dump(test_dict, f)
    else:
        img = cv2.imread(LOW_LIGHT_PATH + '/1.png')
        cv2.imwrite('images/1_original.png', img)
        H, W, _ = img.shape # (400, 600, 3)
        rev_img = 255 - img
        cv2.imwrite('images/1_reverse.png', rev_img)
        dark_channel = esti_dark_channel(rev_img, H, W, D)
        cv2.imwrite('images/1_dark_channel.png', dark_channel)
        atmospheric_light = esti_atmos_light(rev_img, dark_channel, H, W)
        transmission = esti_trans(rev_img, atmospheric_light, H, W, OMEGA, D)
        img_no_frog = (rev_img.astype(np.int32) - atmospheric_light.astype(np.int32)) \
                                    / np.expand_dims(np.clip(transmission, T0, 1.0), -1) + atmospheric_light
        img_no_frog = np.clip(img_no_frog, 0, 255).astype(np.uint8)
        img_enhance = 255 - img_no_frog
        cv2.imwrite('images/1_no_frog.png', img_no_frog)
        cv2.imwrite('images/1_no_frog_dc.png', esti_dark_channel(img_no_frog, H, W, D))
        cv2.imwrite('images/1_enhanced.png', img_enhance)
        imgHigh = cv2.imread(HIGH_LIGHT_PATH + '/1.png')
        psnr = peak_signal_noise_ratio(imgHigh, img_enhance)
        ssim = structural_similarity(imgHigh, img_enhance, channel_axis=2)
    print('PSNR:', psnr)
    print('SSIM:', ssim)
    