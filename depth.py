# import cv2
# import torch

# from depth_anything_v2.dpt import DepthAnythingV2

# DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

# model_configs = {
#     'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
#     'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
#     'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
#     'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
# }

# encoder = 'vitl' # or 'vits', 'vitb', 'vitg'

# model = DepthAnythingV2(**model_configs[encoder])
# model.load_state_dict(torch.load(f'checkpoints/depth_anything_v2_{encoder}.pth', map_location='cpu'))
# model = model.to(DEVICE).eval()

# raw_img = cv2.imread('/home/lwj/data/TAT-GS/image_llff.jpg')
# depth = model.infer_image(raw_img) # HxW raw depth map in numpy

from transformers import pipeline
from PIL import Image
import numpy as np

# pipe = pipeline(task="depth-estimation", model="LiheYoung/depth-anything-small-hf")
pipe = pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf")
image = Image.open('/home/lwj/data/TAT-GS/image_llff.jpg')
depth = pipe(image)["depth"]
# 将图像转换为 NumPy 数组
depth_array = np.array(depth)

# 反转深度图：可以通过减去最大值来实现
inverted_depth_array = 255 - depth_array

# 将反转后的数组转换回图像
inverted_depth_image = Image.fromarray(inverted_depth_array)
inverted_depth_image.save("v2depth.png")

