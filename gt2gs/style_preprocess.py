# style_preprocess.py
from dataclasses import dataclass 
import torch 
from gs.gaussian_renderer import render
from gt2gs.style_utils import *
import os
import time
from icecream import ic
import torchvision
import numpy as np
from gs.utils.general_utils import inverse_sigmoid
from torch import nn
import os
from simple_knn._C import distCUDA2
from gs.scene.gaussian_model import GaussianModel
from gt2gs.style_utils import render_depth_or_mask_images
from gt2gs.style_loss import FeatureExtractor
import math
from scipy.fftpack import dct

def compute_frequency_density_from_chw_tensor(image_tensor: torch.Tensor) -> torch.Tensor:
    """
    输入:
        image_tensor: torch.Tensor, 形状为 (C, H, W)，像素已归一化在 [0, 1]
    输出:
        torch.Tensor, 形状为 (H//8, W//8)，归一化后的频率密度图
    """
    assert isinstance(image_tensor, torch.Tensor)
    assert image_tensor.ndim == 3, "图像必须是 [C, H, W] 形式"
    C, H, W = image_tensor.shape
    assert H % 8 == 0 and W % 8 == 0, "图像尺寸必须是8的倍数"

    # 转灰度（若已有灰度图则跳过）
    if C == 3:
        r, g, b = image_tensor[0], image_tensor[1], image_tensor[2]
        gray = 0.299 * r + 0.587 * g + 0.114 * b
    elif C == 1:
        gray = image_tensor[0]
    else:
        raise ValueError("仅支持输入通道为 1 或 3 的图像")

    # 转 numpy 并恢复像素范围
    gray_np = gray.detach().cpu().numpy().astype(np.float32) * 255.0
    gray_np -= 128.0  # DCT 预处理：中心化

    # 分块：转为形状 (H//8, 8, W//8, 8) → (h_blocks, w_blocks, 8, 8)
    blocks = gray_np.reshape(H // 8, 8, W // 8, 8).transpose(0, 2, 1, 3)

    # 计算 DCT
    dct_blocks = dct(dct(blocks, axis=-2, norm='ortho'), axis=-1, norm='ortho')

    # 提取高频区域（右下 4×4）并计算能量
    high_freq = dct_blocks[:, :, 4:, 4:]
    energy_map = np.sum(np.abs(high_freq), axis=(-2, -1))  # shape = (H//8, W//8)

    # 归一化为 [0, 1]
    norm_map = (energy_map - energy_map.min()) / (energy_map.ptp() + 1e-8)
    fre_map = torch.tensor(norm_map, dtype=torch.float32)
    # fre_map = 1.0 - fre_map

    return fre_map

def _init_depth_images(trainer):
    
    viewpoint_stack = trainer.scene.getTrainCameras()
    trainer.ctx.depth_images = []
    for _, view in enumerate(viewpoint_stack):
        depth_image = trainer.get_render_pkgs(view)["depth"]
        trainer.ctx.depth_images.append(depth_image.squeeze().detach())
        
    trainer.ctx.depth_images = torch.stack(trainer.ctx.depth_images).to(device=trainer.device)

def _init_frequency_images(trainer):
    viewpoint_stack = trainer.scene.getTrainCameras()
    trainer.ctx.frequency_images = []
    for _, view in enumerate(viewpoint_stack):
        image = view.original_image
        _, H, W = image.shape
        H_crop = (H // 8) * 8  #
        W_crop = (W // 8) * 8  #
        frequency_images = compute_frequency_density_from_chw_tensor(image[:,:H_crop, :W_crop])
        trainer.ctx.frequency_images.append(frequency_images.squeeze().detach())
        concat_and_save_images("./fre_image.jpg", frequency_images.squeeze().detach())
    trainer.ctx.frequency_images = torch.stack(trainer.ctx.frequency_images).to(device=trainer.device)


        
def _init_scene_images(trainer):
    
    viewpoint_stack = trainer.scene.getTrainCameras()
    trainer.ctx.scene_images = []
    
    # colmap maybe change image's size
    min_h, min_w = 10000, 10000
    for i, view in enumerate(viewpoint_stack):
        min_h = min(min_h, view.image_height)
        min_w = min(min_w, view.image_width)
    trainer.ctx.image_width = min_w
    trainer.ctx.image_height = min_h
    
    for _, view in enumerate(viewpoint_stack):
        trainer.ctx.scene_images.append(view.original_image[:, :min_h, :min_w])
        
    trainer.ctx.scene_images = torch.stack(trainer.ctx.scene_images).to(device=trainer.device)
        
    
def _init_style_images(trainer):
    
    trainer.ctx.style_image = read_and_resize_image(
        trainer.config.style.style_image, 
        trainer.config.style.style_image_size,
    ).to(device=trainer.device).contiguous()

# TODO: frequency
def _init_depth_group(trainer, depth_images):
    
    depth_group_num = trainer.config.style.depth_group_num
    depth_group_interval = math.ceil(256 / depth_group_num)
    
    # ic(depth_group_num, depth_group_interval)
    
    depth_masks = torch.zeros_like(depth_images, device=trainer.device)
    
    for i, depth_image in enumerate(depth_images):
        normalized_depth_image = normalize_depth_to_uint8(depth_image)
        
        for j in range(depth_group_num):
            # [l_point, r_point]
            l_point = depth_group_interval * j
            r_point = min(255, depth_group_interval * (j + 1) - 1)
            
            mask = torch.logical_and(normalized_depth_image >= l_point, 
                                     normalized_depth_image <= r_point)
            
            depth_masks[i][mask] = j
        # ic(scale_value[i])
        # for j in range(len(scale_value[i]) - 1, -1, -1):
        #     # ic(scale_value[i][j], scale_value[i][0])
        #     scale_value[i][j] /= scale_value[i][0]
            # ic(scale_value[i][j])
        # exit(0)
    # exit(0)
    # for i, k in enumerate(depth_masks):
    #     ic(i, k.shape)
    
    return depth_masks

# def _init_frequency_group(trainer, frequency_images):
    
#     # TODO: add trainer.config.style.frequency_group_num
#     frequency_group_num = trainer.config.style.frequency_group_num
#     # TODO: find the max frequency (not 256)
#     frequency_group_interval = 1.0 / frequency_group_num
    
#     # ic(depth_group_num, depth_group_interval)
    
#     frequency_masks = torch.zeros_like(frequency_images, device=trainer.device)
    
#     for i, frequency_image in enumerate(frequency_images):
#         normalized_depth_image = normalize_depth_to_uint8(depth_image)
        
#         for j in range(depth_group_num):
#             # [l_point, r_point]
#             l_point = depth_group_interval * j
#             r_point = min(255, depth_group_interval * (j + 1) - 1)
            
#             mask = torch.logical_and(normalized_depth_image >= l_point, 
#                                      normalized_depth_image <= r_point)
            
#             depth_masks[i][mask] = j
#         # ic(scale_value[i])
#         # for j in range(len(scale_value[i]) - 1, -1, -1):
#         #     # ic(scale_value[i][j], scale_value[i][0])
#         #     scale_value[i][j] /= scale_value[i][0]
#             # ic(scale_value[i][j])
#         # exit(0)
#     # exit(0)
#     # for i, k in enumerate(depth_masks):
#     #     ic(i, k.shape)
        
#     return depth_masks

def _init_style_downscaling(trainer, style_image):
    
    _, h, w = style_image.shape
    downscaling_num = trainer.config.style.depth_group_num
    downscale_ratio = torch.linspace(1, trainer.config.style.downscale_limit_ratio, downscaling_num)
    # ic(downscale_ratio)
    # exit(0)
    
    downscaled_style_images_list = []
    for i in range(downscaling_num):
        new_h = int(h / downscale_ratio[i])
        new_w = int(w / downscale_ratio[i])
        
        # ic(new_h)
        
        downscaled_style_images_list.append(F.interpolate(style_image.unsqueeze(0),
                                                         size=(new_h, new_w),
                                                         mode='bilinear',
                                                         align_corners=False,
                                                         antialias=True).squeeze(0))
    # exit(0)
    return downscaled_style_images_list

def _init_style_features(trainer, style_image_list):
    style_features_list = []
    style_matrix_list = []
    
    for i, style_image in enumerate(style_image_list):
        
        if trainer.config.style.enable_feature_enhancement:
            style_features, style_matrix = get_enhanced_style_features(trainer, style_image)
        else:
            style_features, style_matrix = get_original_style_features(trainer, style_image)
            
        style_features_list.append(style_features)
        style_matrix_list.append(style_matrix)
        
    return style_features_list, style_matrix_list

# TODO: new mask list
def _init_scene_features(trainer, scene_images, masks):
    depth_group_num = trainer.config.style.depth_group_num

    scene_features_list = []
    scene_features_mask_list = []
    for i, scene_image in enumerate(scene_images):
        features = trainer.feature_extractor(scene_image)
        # ic(masks[i].shape)
        downscaled_masks = labels_downscale(masks[i], features.shape[-2:])
        features_list = get_separated_list(features, downscaled_masks, depth_group_num)
        scene_features_list.append(features_list)
        scene_features_mask_list.append(downscaled_masks)
    
    return scene_features_list, scene_features_mask_list

@torch.no_grad
def preprocess(trainer):
    trainer.ctx = StyleContext()
    _init_scene_images(trainer)
    _init_depth_images(trainer)
    _init_style_images(trainer)
    _init_frequency_images(trainer)
    
    # TODO: (done?) add frequency group
    depth_masks = _init_depth_group(trainer, trainer.ctx.depth_images)
    # frequency_masks = _init_frequency_group(trainer, trainer.ctx.scene_images)


    # for i, v in enumerate(scale_value):
    #     ic(v)
    # for i, depth_mask in enumerate(depth_masks):
    #     render_depth_or_mask_images(f"./debug/depth_mask/{i}.jpg", depth_mask)
    # exit(0)
    
    start_time_GTA = time.perf_counter()
    
    downscaled_style_images_list = _init_style_downscaling(trainer, trainer.ctx.style_image)
    # for i, image in enumerate(downscaled_style_images_list):
    #     # ic(image.shape)
    #     render_RGBcolor_images(f"./debug/downscaled_images/{i}.jpg", image)
    if trainer.config.style.enable_color_transfer:
        color_transfer(trainer.ctx)
    
    trainer.feature_extractor = FeatureExtractor()
    style_features_list, style_matrix_list = _init_style_features(trainer, downscaled_style_images_list)
    end_time_GTA = time.perf_counter()
    elapsed_time_GTA = end_time_GTA - start_time_GTA
    print(f"Time for GTA: {elapsed_time_GTA:.4f} seconds")
    
    # for i, style_features in enumerate(style_features_list):
    #     ic(style_features.shape)
    
    # TODO: depth mask from here, will add frequency mask
    scene_features_list, scene_features_mask_list = _init_scene_features(trainer, trainer.ctx.scene_images, depth_masks)
    # for i, features_list in enumerate(scene_features_list):
    #     for j, features in enumerate(features_list):
    #         ic(i, j, features.shape)
    

    trainer.warper = Warper()
    trainer.ctx.style_features_list = style_features_list
    trainer.ctx.style_matrix_list = style_matrix_list
    
    trainer.ctx.scene_masks = depth_masks
    trainer.ctx.scene_features_list = scene_features_list
    trainer.ctx.scene_features_mask_list = scene_features_mask_list

    trainer.ctx.fusion_masks = []
    for i, fre_image in enumerate(trainer.ctx.frequency_images):
        fusion = 2.0 - fre_image - scene_features_mask_list[i] / (trainer.config.style.depth_group_num - 1)
        trainer.ctx.fusion_masks.append(fusion.detach())
    
    a = 1



    # TODO: add fusion mask
    
    
    # _init_add_gaussians(trainer)
    
    
    
    
    
        
    
        
    