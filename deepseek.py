from PIL import Image
import numpy as np

# 打开RGBA图像
img1 = Image.open('/data3/lwj/original_data/blender/lego/test/r_0.png')

# 创建白色背景图像
background = Image.new('RGB', img1.size, (255, 255, 255))  # 白色背景

# 将背景图像转换为RGBA模式
background_rgba = background.convert('RGBA')

# 将背景图像与原图像进行合成，处理透明区域
img1 = Image.alpha_composite(background_rgba, img1)

# 将图像转换为RGB模式
img1 = img1.convert('RGB')
image_curr = np.array(img1)
W, H = img1.size

# 确保cx和cy正确
cx = W / 2
cy = H / 2


# 打开深度图并转换为灰度图
# depth_curr = Image.open('/data3/lwj/original_data/blender/lego/test/r_0.png').convert('L')
depth_curr = Image.open('/data3/lwj/original_data/blender/lego/test/r_0_depth_0001.png').convert('L')
depth_array = np.array(depth_curr).astype(np.float32)
# depth_array = (255 - depth_array) / 255.0  # 归一化到[0,1]

# 将深度值转换为实际距离
# near = 0.3
far = 6
z_values = far * (255 - depth_array) / 255
# z_values = near * far / far - (far - near) * depth_array
depth1= z_values.astype(np.float32)


import numpy as np

fx = 1134.4
fy = 1134.4

# 内参矩阵 K
K = np.array([
    [1134.4, 0, cx],
    [0, 1134.4, cy],
    [0, 0, 1]
], dtype=np.float32)

intrinsic1 = K
intrinsic2 = K

# 旋转矩阵 R0
R0 = np.array([
    [-0.9999999403953552, 0.0, 0.0],
    [0.0, -0.7341099977493286, 0.6790305972099304],
    [0.0, 0.6790306568145752, 0.7341098785400391]
], dtype=np.float32)

# 平移向量 T0
T0 = np.array([
    [0.0],
    [2.737260103225708],
    [2.959291696548462]
], dtype=np.float32)

transformation1 = np.array([
    [-0.9999999403953552, 0.0, 0.0, 0.0],
    [0.0, -0.7341099977493286, 0.6790305972099304, 2.737260103225708],
    [0.0, 0.6790306568145752, 0.7341098785400391, 2.959291696548462],
    [0.0, 0.0, 0.0, 1.0]
], dtype=np.float32)

# 旋转矩阵 R
R = np.array([
    [-0.9048271179199219, 0.3107704222202301, -0.2910493314266205],
    [-0.4257793426513672, -0.6604207754135132, 0.6185113191604614],
    [0.0, 0.6835686564445496, 0.7298861742019653]
], dtype=np.float32)

# 平移向量 T
T = np.array([
    [-1.1732574701309204],
    [2.4932990074157715],
    [2.942265510559082]
], dtype=np.float32)

transformation2 =np.array([
    [-0.9048271179199219, 0.3107704222202301, -0.2910493314266205, -1.1732574701309204],
    [-0.4257793426513672, -0.6604207754135132, 0.6185113191604614, 2.4932990074157715],
    [0.0, 0.6835686564445496, 0.7298861742019653, 2.942265510559082],
    [0.0, 0.0, 0.0, 1.0]
], dtype=np.float32)

transformation = np.matmul(transformation2, np.linalg.inv(transformation1))
# # 定义旋转矩阵R和平移向量T
# R = np.array([
#     [1.0, 0.0, 0.0],
#     [0.0, 1.0, 0.0],
#     [0.0, 0.0, 1.0]
# ], dtype=np.float32)

# T = np.array([
#     [1.874],
#     [-1.0],
#     [2.286]
# ], dtype=np.float32)

# # 生成网格坐标
# x, y = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing='xy')
# ones_2d = np.ones(shape=(H, W), dtype=np.float32)

y1d = np.array(range(H))
x1d = np.array(range(W))
x2d, y2d = np.meshgrid(x1d, y1d)
ones_2d = np.ones(shape=(H, W))
ones_4d = ones_2d[:, :, None, None]
pos_vectors_homo = np.stack([x2d, y2d, ones_2d], axis=2)[:, :, :, None]

intrinsic1_inv = np.linalg.inv(intrinsic1)
intrinsic1_inv_4d = intrinsic1_inv[None, None]
intrinsic2_4d = intrinsic2[None, None]
depth_4d = depth1[:, :, None, None]
trans_4d = transformation[None, None]

unnormalized_pos = np.matmul(intrinsic1_inv_4d, pos_vectors_homo)
world_points = depth_4d * unnormalized_pos
world_points_homo = np.concatenate([world_points, ones_4d], axis=2)
trans_world_homo = np.matmul(trans_4d, world_points_homo)
trans_world = trans_world_homo[:, :, :3]
trans_norm_points = np.matmul(intrinsic2_4d, trans_world)
trans_points1 = trans_norm_points
trans_coordinates = trans_points1[:, :, :2, 0] / trans_points1[:, :, 2:3, 0]
trans_depth1 = trans_points1[:, :, 2, 0]

# 筛选出在有效范围内的索引
x_coords = pixel_coord_cam2[:, 0]
y_coords = pixel_coord_cam2[:, 1]
valid_idx = np.where(
    (x_coords >= 0) & (x_coords < W) &  
    (y_coords >= 0) & (y_coords < H)    
)[0]

# 获取有效点的颜色
valid_colors = image_curr.reshape(-1, 3)[valid_idx]

# 生成新的图像
image2 = np.zeros((H, W, 3), dtype=np.float32)
x_valid = x_coords[valid_idx].astype(int)
y_valid = y_coords[valid_idx].astype(int)
image2[y_valid, x_valid] = valid_colors

# 确保图像值在[0,255]之间
image2 = np.clip(image2, 0, 255)
image2_uint8 = image2.astype(np.uint8)

# 保存图像
image_pil = Image.fromarray(image2_uint8)
image_pil.save("blender.png")