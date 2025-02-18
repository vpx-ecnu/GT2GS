#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
import cv2
import numpy as np
from utils.camera_utils import compute_epipolar_projection
from torchvision.transforms import ToPILImage
import torch.nn.functional as F
from PIL import Image
from scipy.interpolate import griddata as interp_grid
from transformers import pipeline
from PIL import Image

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

def render_RGBcolor_images(path, image):
    """
    Renders and saves RGB color images.
    
    @param path: The path to save the image.
    @param image: The tensor image to render. Shape: [3, H, W]
    """
    image = image.detach().permute(1, 2, 0).clamp(min=0.0, max=1.0).cpu().numpy()
    image = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, image)

def render_depth_or_mask_images(path, image):
    """
    Renders and saves depth or mask images.
    
    @param path: The path to save the image.
    @param image: The tensor image to render. Shape: [1, H, W]
    """
    image = image.detach().cpu().numpy().squeeze()
    depth_map_normalized = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
    depth_map_normalized = np.uint8(depth_map_normalized)
    # depth_map_colored = cv2.applyColorMap(depth_map_normalized, cv2.COLORMAP_JET)
    cv2.imwrite(path, depth_map_normalized)

def save_tensor_as_image(tensor, filepath):
    """
    保存Tensor为图片文件。
    Args:
        tensor: [B, C, H, W] 的Tensor，像素值应在 [0, 1] 范围。
        filepath: 保存图片的路径。
    """
    # 确保 tensor 在 [0, 1] 范围内
    tensor = tensor.clamp(0, 1)
    # 移除批次维度并转为 NumPy
    image_array = (tensor[0].permute(1, 2, 0).cpu().numpy() * 255).astype('uint8')  # [H, W, C]
    # 保存为图片
    image = Image.fromarray(image_array)
    image.save(filepath)
    return image
    
def save_image(img_tensor, file_path):
    """
    将 PyTorch 图像张量保存为图片文件。
    
    Args:
        img_tensor (torch.Tensor): 输入的图像张量，形状为 (C, H, W)，值范围为 [0, 1] 或 [0, 255]。
        file_path (str): 保存的文件路径，包括文件名和扩展名。
    """
    # 确保图像张量在 CPU 上，且类型为浮点数
    img_tensor = img_tensor.cpu().float()
    
    # 如果张量值范围是 [0, 1]，需要缩放到 [0, 255]
    if img_tensor.max() <= 1.0:
        img_tensor = img_tensor * 255.0
    
    # 转换为 PIL 图像
    pil_image = ToPILImage()(img_tensor.byte())
    
    # 保存为图片文件
    pil_image.save(file_path)
    print(f"Image saved to {file_path}")
    return pil_image

def pixel_to_3d(depth_map, K_inv):
    B, H, W = depth_map.shape
    y, x = torch.meshgrid(torch.arange(H, device=depth_map.device), torch.arange(W, device=depth_map.device), indexing="ij")
    x, y = x.float(), y.float()
    z = depth_map.view(B, -1)

    ones = torch.ones_like(x).view(1, -1).expand(B, -1).to(depth_map.device)
    pixel_coords = torch.stack([x.flatten(), y.flatten(), ones.flatten()], dim=0).to(depth_map.device)  # [3, H*W]

    points_3d = (K_inv @ pixel_coords).permute(1, 0).view(B, H, W, 3) * depth_map.unsqueeze(-1)  # [B, H, W, 3]
    return points_3d

def project_3d_to_pixel(points_3d, cam1, cam2, K):
    B, H, W, _ = points_3d.shape
    points_3d_h = torch.cat([points_3d, torch.ones_like(points_3d[..., :1])], dim=-1)  # [B, H, W, 4]

    # 从源相机坐标系转换到目标相机坐标系
    transform = cam2.world_view_transform @ torch.inverse(cam1.world_view_transform)  # 相机间变换矩阵
    points_3d_cam2 = torch.einsum("ij,bhwj->bhwi", transform, points_3d_h)  # [B, H, W, 4]

    # 使用内参矩阵进行投影
    points_2d_h = torch.einsum("ij,bhwj->bhwi", K, points_3d_cam2[..., :3])  # [B, H, W, 3]
    points_2d = points_2d_h[..., :2] / points_2d_h[..., 2:3]  # [B, H, W, 2]
    return points_2d

def sample_pixels(img1, points_2d):
    
    if img1.dim() == 3:  # 如果图像没有 batch 维度，添加一个
        img1 = img1.unsqueeze(0)
    B, H, W, _ = points_2d.shape
    points_2d_normalized = points_2d.clone()
    points_2d_normalized[..., 0] = (points_2d[..., 0] / (W - 1)) * 2 - 1  # x 坐标归一化到 [-1, 1]
    points_2d_normalized[..., 1] = (points_2d[..., 1] / (H - 1)) * 2 - 1  # y 坐标归一化到 [-1, 1]

    sampled_img = sampled_img = F.grid_sample(img1, points_2d_normalized, mode='bilinear', padding_mode='zeros', align_corners=True)  # [B, C, H, W]
    return sampled_img.squeeze(1)

def run_code():

    img1 = Image.open('./junwei_data/left.png')
    image_curr = np.array(img1)
    W, H = img1.size
    
    fx = fy = 415.7
    cx = W / 2
    cy = H / 2
    K = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ], dtype=np.float32)
    depth_curr = Image.open('./junwei_data/left_depth.png').convert('L')
    depth_array = np.array(depth_curr) / 255.0
    print("Depth array shape:", depth_array.shape)

    # 将深度值转换为实际距离
    near = 0.3
    far = 1000
    z_values = far - (far - near) * depth_array
    z_values = z_values
    depth_curr = z_values.astype(np.float32)
    
    
    R0 = np.array([
        [0.9703, 0, 0.2419],
        [0, 1, 0],
        [-0.2419, 0, 0.9703]
    ], dtype=np.float32)

    T0 = np.array([
        [-2.233],
        [1],
        [-2.235]
    ], dtype=np.float32)
    
    img2 = Image.open('./junwei_data/right.png')
    # 旋转矩阵R
    R = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0]
    ], dtype=np.float32)

    # 平移向量T
    T = np.array([
        [1.874],
        [-1.0],
        [2.286]
    ], dtype=np.float32)
    
    x, y = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing='xy')
    y = -y
    grid = np.stack((x,y), axis=-1).reshape(-1,2)
    pts_coord_cam = np.matmul(np.linalg.inv(K), np.stack((x*depth_curr, y*depth_curr, 1*depth_curr), axis=0).reshape(3,-1))
    new_pts_coord_world2 = (np.linalg.inv(R0).dot(pts_coord_cam) - np.linalg.inv(R0).dot(T0)).astype(np.float32) ## new_pts_coord_world2
    new_pts_colors2 = (np.array(image_curr).reshape(-1,3)) ## new_pts_colors2
    pts_coord_world, pts_colors = new_pts_coord_world2.copy(), new_pts_colors2.copy()

    ### Transform world to pixel
    pts_coord_cam2 = R.dot(pts_coord_world) + T  ### Same with c2w*world_coord (in homogeneous space)
    # 已经验证了到这里的结果是正确的（逆变换回来）
    qwq = pts_coord_cam2 - pts_coord_cam
    pixel_coord_cam2 = np.matmul(K, pts_coord_cam2) #.reshape(3,H,W).transpose(1,2,0).astype(np.float32)
    pixel_coord_cam2 = pixel_coord_cam2[:2, :] / pixel_coord_cam2[2, :]
    pixel_coord_cam2 = pixel_coord_cam2.transpose(1, 0)
    x_coords = pixel_coord_cam2[:, 0]  # 获取所有的 x 坐标
    y_coords = pixel_coord_cam2[:, 1]  # 获取所有的 y 坐标

    # 筛选出在有效范围内的索引
    valid_idx = np.where(
        (x_coords >= 0) & (x_coords < W) &  # x 坐标在 [0, W-1] 范围内
        (y_coords >= 0) & (y_coords < H)    # y 坐标在 [0, H-1] 范围内
    )[0]

    x, y = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing='xy')
    grid = np.stack((x,y), axis=-1).reshape(-1,2)
    image2 = interp_grid(pixel_coord_cam2[valid_idx], pts_colors[valid_idx], grid, method='linear', fill_value=0).reshape(H,W,3)
    image2 = np.clip(image2, 0, 255)  # 确保图像值在 [0, 1] 之间
    image2_uint8 = image2.astype(np.uint8)
    # 将其转换为 0-255 之间的整数值，以符合图片格式的要求
    # image2_uint8 = (image2 * 255).astype(np.uint8)

    # 保存图片
    image_pil = Image.fromarray(image2_uint8)
    image_pil.save("junwei_result.png")

if __name__ == "__main__":
    run_code()