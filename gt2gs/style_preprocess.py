# style_preprocess.py
from dataclasses import dataclass 
import torch 
from gs.gaussian_renderer import render
from gt2gs.style_utils import *
import os
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

def _init_depth_images(trainer):
    
    viewpoint_stack = trainer.scene.getTrainCameras()
    trainer.ctx.depth_images = []
    for _, view in enumerate(viewpoint_stack):
        depth_image = trainer.get_render_pkgs(view)["depth"]
        trainer.ctx.depth_images.append(depth_image.squeeze().detach())
        
    trainer.ctx.depth_images = torch.stack(trainer.ctx.depth_images).to(device=trainer.device)
        
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

def _init_add_gaussians(trainer):
    
    def edge_detection(image):
        gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        gray_image = (gray_image * 255).astype(np.uint8)
        edges = cv2.Canny(gray_image, threshold1=50, threshold2=150)

        window_size = 15
        edge_densify = cv2.blur(edges, (window_size, window_size))

        threshold = 1
        return edge_densify < threshold
    
    init_image_idx = torch.arange(0, trainer.ctx.scene_images.shape[0], 
                                  trainer.config.style.init_densification_image_intervals)
    
    original_gaussian_points = trainer.gaussians.get_xyz
    original_gaussian_nums = original_gaussian_points.shape[0]
    
    new_gaussian_points = []
    new_gaussian_colors = []
    
    for image_idx in init_image_idx:
        
        image = trainer.ctx.scene_images[image_idx]
        image_np = image.permute(1, 2, 0).detach().cpu().numpy()
        
        # low_texture_mask = torch.logical_not(torch.tensor(edge_detection(image_np), device=trainer.device))
        low_texture_mask = torch.tensor(edge_detection(image_np), device=trainer.device)
        
        mask = torch.zeros_like(low_texture_mask, dtype=torch.bool)
        mask[::trainer.config.style.init_densification_downsample, ::trainer.config.style.init_densification_downsample] = True
        low_texture_mask = low_texture_mask & mask
        # render_depth_or_mask_images(f"./debug/{image_idx}_downsample.jpg", low_texture_mask.int())
        # continue

        # unproject
        depth = trainer.ctx.depth_images[image_idx]
        _, h, w = image.shape
        
        x1d = torch.arange(0, w)[None]
        y1d = torch.arange(0, h)[:, None]
        x2d = x1d.repeat([h, 1]).to(depth)  # (h, w)
        y2d = y1d.repeat([1, w]).to(depth)  # (h, w)
        ones_2d = torch.ones(size=(h, w)).to(depth)  # (h, w)
        
        pos_vectors_homo = torch.stack([x2d, y2d, ones_2d], dim=2)[:, :, :, None]  # (h, w, 3, 1)

        intrinsic1_inv = torch.linalg.inv(torch.tensor(trainer.scene.getTrainCameras()[image_idx].K, device=trainer.device))  # (3, 3)
        intrinsic1_inv_4d = intrinsic1_inv[None, None, :]  # (1, 1, 3, 3)
        depth_4d = depth[:, :, None, None]  # (h, w, 1, 1)
        
        # ic(intrinsic1_inv_4d.shape)
        # ic(pos_vectors_homo.shape)

        unnormalized_pos = torch.matmul(intrinsic1_inv_4d, pos_vectors_homo)  # (h, w, 3, 1)
        
        # ic(depth_4d.shape)
        # ic(unnormalized_pos.shape)
        
        world_points = depth_4d * unnormalized_pos  # (h, w, 3, 1)
        world_points = world_points[low_texture_mask].squeeze()
        
        new_gaussian_colors.append(image[:, low_texture_mask].T)
        new_gaussian_points.append(world_points)
        break
        
        
    new_gaussian_colors = torch.cat(new_gaussian_colors)
    new_gaussian_points = torch.cat(new_gaussian_points)
    
    ic(new_gaussian_colors.shape)
    ic(new_gaussian_points.shape)
    
    new_gaussian = GaussianModel(trainer.gaussians.max_sh_degree)
    
    features = torch.zeros((new_gaussian_colors.shape[0], 3, (new_gaussian.max_sh_degree + 1) ** 2)).float().cuda()
    features[:, :3, 0 ] = new_gaussian_colors
    features[:, 3:, 1:] = 0.0


    dist2 = torch.clamp_min(distCUDA2(new_gaussian_points), 0.0000001)
    scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
    rots = torch.zeros((new_gaussian_points.shape[0], 4), device="cuda")
    rots[:, 0] = 1

    opacities = inverse_sigmoid(0.1 * torch.ones((new_gaussian_points.shape[0], 1), dtype=torch.float, device="cuda"))

    new_gaussian._xyz = nn.Parameter(new_gaussian_points.requires_grad_(True))
    new_gaussian._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
    new_gaussian._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
    new_gaussian._scaling = nn.Parameter(scales.requires_grad_(True))
    new_gaussian._rotation = nn.Parameter(rots.requires_grad_(True))
    new_gaussian._opacity = nn.Parameter(opacities.requires_grad_(True))
    new_gaussian.max_radii2D = torch.zeros((new_gaussian.get_xyz.shape[0]), device="cuda")
    
    
    point_cloud_path = os.path.join(trainer.config.style.stylized_model_path, "point_cloud/iteration_{}".format(1))
    new_gaussian.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))
    
    # new_gaussian_colors = torch.cat(new_gaussian_colors)
    # new_gaussian_points = torch.cat(new_gaussian_points)
    # # ic(new_gaussian_points.requires_grad)
    
    # # ic(new_gaussian_colors.shape)
    # # ic(new_gaussian_points.shape)
    
    # dist2 = torch.clamp_min(distCUDA2(new_gaussian_points), 0.0000001)
    # scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
    # rots = torch.zeros((new_gaussian_points.shape[0], 4), device=trainer.device)
    # rots[:, 0] = 1

    # # ic(trainer.gaussians.get_features.shape)
    
    # new_xyz = new_gaussian_points.requires_grad_()
    # new_features_dc = new_gaussian_colors[:, None, :].requires_grad_()
    # new_features_rest = torch.zeros((new_gaussian_colors.shape[0], (trainer.gaussians.max_sh_degree + 1) ** 2 - 1, 3), 
    #                                  dtype=torch.float32, device=trainer.device, requires_grad=True)
    # new_opacity = inverse_sigmoid(0.5 * torch.ones((new_gaussian_points.shape[0], 1), dtype=torch.float, device=trainer.device, requires_grad=True))
    # new_scaling = scales.requires_grad_()
    # new_rotation = rots.requires_grad_()
    
    # # ic(new_xyz.shape)
    # # ic(new_features_dc.shape)
    # # ic(new_features_rest.shape)
    # # ic(new_opacity.shape)
    # # ic(new_scaling.shape)
    # # ic(new_rotation.shape)
    
    
    # trainer.gaussians.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation)
    
    # # ic(trainer.config.model.model_path)
    # trainer.scene.save(1, trainer.config.model.model_path)
    
    
    # print(f"Initialize new points: {new_gaussian_points.shape[0]}")
    # point_cloud_path = os.path.join(trainer.config.model.model_path, "point_cloud/iteration_{}".format(1))
    # new_gaussian.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))
    # exit(0)
    exit(0)        

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
        
        style_features, style_matrix = get_enhanced_style_features(trainer, style_image)
        style_features_list.append(style_features)
        style_matrix_list.append(style_matrix)
        
    return style_features_list, style_matrix_list

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
    
    depth_masks = _init_depth_group(trainer, trainer.ctx.depth_images)
    # for i, v in enumerate(scale_value):
    #     ic(v)
    # for i, depth_mask in enumerate(depth_masks):
    #     render_depth_or_mask_images(f"./debug/depth_mask/{i}.jpg", depth_mask)
    # exit(0)
    
    downscaled_style_images_list = _init_style_downscaling(trainer, trainer.ctx.style_image)
    # for i, image in enumerate(downscaled_style_images_list):
    #     # ic(image.shape)
    #     render_RGBcolor_images(f"./debug/downscaled_images/{i}.jpg", image)
    if trainer.config.style.color_transfer:
        color_transfer(trainer.ctx)
    
    trainer.feature_extractor = FeatureExtractor()
    style_features_list, style_matrix_list = _init_style_features(trainer, downscaled_style_images_list)
    
    # for i, style_features in enumerate(style_features_list):
    #     ic(style_features.shape)
    
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
    
    
    # _init_add_gaussians(trainer)
    
    
    
    
    
        
    
        
    