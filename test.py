import cv2
import numpy as np

# 读取图像并转为灰度图
image = cv2.imread('/Datasets/preprocessed_data/llff/horns/images/DJI_20200223_163016_842.png', cv2.IMREAD_GRAYSCALE)

# 边缘检测
edges = cv2.Canny(image, threshold1=50, threshold2=150)

# 计算边缘密度
window_size = 15
edge_density = cv2.blur(edges, (window_size, window_size))

# 设定阈值
threshold = 1
low_texture_mask = edge_density < threshold

print(np.sum(low_texture_mask > 0), np.sum(low_texture_mask == 0))

# f = np.fft.fft2(image)
# fshift = np.fft.fftshift(f)
# magnitude_spectrum = np.log(np.abs(fshift))

# # 设定低频阈值
# threshold = 5
# low_texture_mask = magnitude_spectrum < threshold

# low_texture_mask = np.fft.ifftshift(low_texture_mask)
# low_texture_mask = np.fft.ifft2(low_texture_mask).real
# data = low_texture_mask
# normalized_data = (data - np.min(data)) / (np.max(data) - np.min(data))
# print(normalized_data)
# print((normalized_data * 255).astype(np.uint8)[:, :, None])
# x = (normalized_data * 255).astype(np.uint8)[:, :, None]
# print(x.shape)
cv2.imwrite("11.jpg", (low_texture_mask * 255).astype(np.uint8))

# 显示结果
# cv2.imshow('Low Texture Regions', )
# cv2.waitKey(0)