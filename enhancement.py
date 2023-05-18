import os
import cv2
import numpy as np
from tqdm import trange

LOW_LIGHT_PATH = os.path.join('./images/low')
D = 3 # distance
T0 = 0.1 # threshold
OMEGA = 0.95

def get_bounds(x, Max, distance=D):
    """
    returns the lower and upper bound of the given position with distance
    """
    x_min = max(x - distance, 0)
    x_max = min(x + distance, Max)
    return x_min, x_max

def esti_dark_channel(img, H, W):
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
    
def esti_trans(img, atmospheric_light, H, W):
    tmp = img.astype(np.float64) / atmospheric_light
    transmission = np.zeros((H, W))
    for i in trange(H):
        for j in range(W):
            top, bottom = get_bounds(i, H, D)
            left, right = get_bounds(j, W, D)
            local = tmp[top:bottom, left:right, :]
            transmission[i, j] =  1 - OMEGA * min(np.min(local, axis=(0, 1)))
    return transmission

if __name__ == '__main__':
    img = cv2.imread(LOW_LIGHT_PATH + '/1.png')
    cv2.imwrite('images/1_original.png', img)
    H, W, _ = img.shape # (400, 600, 3)
    img = 255 - img
    cv2.imwrite('images/1_reverse.png', img)
    dark_channel = esti_dark_channel(img, H, W)
    cv2.imwrite('images/1_dark_channel.png', dark_channel)
    atmospheric_light = esti_atmos_light(img, dark_channel, H, W)
    transmission = esti_trans(img, atmospheric_light, H, W)
    img_no_frog = (img.astype(np.int32) - atmospheric_light.astype(np.int32)) \
                                / np.expand_dims(np.clip(transmission, T0, 1.0), -1) + atmospheric_light
    img_no_frog = np.clip(img_no_frog, 0, 255).astype(np.uint8)
    img_enhance = 255 - img_no_frog
    cv2.imwrite('images/1_no_frog.png', img_no_frog)
    cv2.imwrite('images/1_no_frog_dc.png', esti_dark_channel(img_no_frog, H, W))
    cv2.imwrite('images/1_enhanced.png', img_enhance)