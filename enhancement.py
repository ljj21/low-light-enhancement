# 读取 images 文件夹下的图片
import os
import cv2
import numpy as np
from tqdm import trange
def local_region(x, bound, distance):
    """
    Gets the coordinates of the pixel with distance around x
    """
    x_min = max(x - distance, 0)
    x_max = min(x + distance, bound)
    return x_min, x_max


LOW_LIGHT_PATH = os.path.join('./images/low')
img = cv2.imread(LOW_LIGHT_PATH + '/1.png') 
H, W, _ = img.shape # (400, 600, 3)
img = 255 - img
cv2.imwrite('images/1_reverse.png', img)
D = 3 # distance
dark_channel = np.zeros((H, W))
for i in trange(H):
    for j in range(W):
        top, bottom = local_region(i, H, D)
        left, right = local_region(j, W, D)
        local = img[top:bottom, left:right, :]
        dark_channel[i, j] = min(np.min(local, axis=(0, 1)))
cv2.imwrite('images/1_dark_channel.png', dark_channel)
# 估计全局大气光值
# 为 dark_channel 中强度为前 0.1% 的像素对应伪有雾图像中亮度最高处对应的 R,G,B 值
total_pixels = H * W
count = int(total_pixels * 0.001)
index = np.unravel_index(np.argsort(-dark_channel, axis=None)[:count], dark_channel.shape)
# print(index[0].shape)
# print(img[index].shape)
atmospheric_light_index = np.argmax(np.sum(img[index], axis=-1))
atmospheric_light = img[index[0][atmospheric_light_index], index[1][atmospheric_light_index]]
# atmospheric_light = img[index][:, atmospheric_light_index]
# 估计透射率
tmp = img.astype(np.float64) / atmospheric_light
omega = 0.95
transmission = np.zeros((H, W))
for i in trange(H):
    for j in range(W):
        top, bottom = local_region(i, H, D)
        left, right = local_region(j, W, D)
        local = tmp[top:bottom, left:right, :]
        transmission[i, j] =  1 - omega * min(np.min(local, axis=(0, 1)))
t0 = 0.1
# 根据透射率和大气光值得到原图像的估计
# print(img.astype(np.int32) - atmospheric_light.astype(np.int32))
img_radiance_estimate = (img.astype(np.int32) - atmospheric_light.astype(np.int32)) / np.expand_dims(np.clip(transmission, t0, 1.0), -1) + atmospheric_light
# print(img_radiance_estimate)
# 限制原图像的估计值在 0-255 之间
img_enhance = 255 - np.clip(img_radiance_estimate, 0, 255)
# 保存原图像的估计值
cv2.imwrite('images/1_enhanced.png', img_enhance)


