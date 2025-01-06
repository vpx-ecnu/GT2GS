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

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):

    first_iter = 0
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, -1, shuffle=False)
    
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)
    
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    bg = torch.rand((3), device="cuda") if opt.random_background else background

    # iter_start = torch.cuda.Event(enable_timing = True)
    # iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack_origin = scene.getTrainCameras()
    for _, view in enumerate(viewpoint_stack_origin):
        depth_image = render(view, gaussians, pipe, bg)["depth"]
        view.depth_image = depth_image.squeeze().detach()
    
    viewpoint_stack = scene.getTrainCameras().copy()
    cam1 = viewpoint_stack.pop()
    render_depth_or_mask_images("debug.png", cam1.depth_image)
    img1 = cam1.original_image.to("cuda")
    # img1_pil = save_image(img1, "lucid_origin_image.png")
    # pipe = pipeline(task="depth-estimation", model="Intel/zoedepth-nyu-kitti")
    # depth_curr = pipe(img1_pil)["depth"]
    # depth_curr.save("qwqdepth.png")
    # depth_curr = np.array(depth_curr).astype(np.float32)
    
    
    
    
    image_curr = cam1.original_image.permute(1,2,0).cpu().numpy()
    cam2 = viewpoint_stack.pop()
    img2 = cam2.original_image.to("cuda")
    H = int(cam1.image_height)
    W = int(cam1.image_width)
    K = cam1.K
    
    x, y = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing='xy')
    grid = np.stack((x,y), axis=-1).reshape(-1,2)
    depth_curr = cam1.depth_image.cpu().numpy()
    # image = depth_curr.squeeze()
    # depth_map_normalized = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
    # # 归一化到 [0, 1] 范围

    # # 如果需要保证数据类型为 float32（或者其他类型），可以转换
    # depth_curr = depth_map_normalized.astype(np.float32)
    
    R0, T0 = cam1.R, cam1.T[:, np.newaxis]
    origin_coord = np.stack((x*depth_curr, y*depth_curr, 1*depth_curr), axis=0).reshape(3,-1)
    # origin_coord = origin_coord[:2, :] / origin_coord[2, :]
    # origin_coord = origin_coord.transpose(1, 0)
    pts_coord_cam = np.matmul(np.linalg.inv(K), np.stack((x*depth_curr, y*depth_curr, 1*depth_curr), axis=0).reshape(3,-1))
    new_pts_coord_world2 = (np.linalg.inv(R0).dot(pts_coord_cam) - np.linalg.inv(R0).dot(T0)).astype(np.float32) ## new_pts_coord_world2
    new_pts_colors2 = (np.array(image_curr).reshape(-1,3)) ## new_pts_colors2
    pts_coord_world, pts_colors = new_pts_coord_world2.copy(), new_pts_colors2.copy()
    
    # img_test = interp_grid(origin_coord, new_pts_colors2, grid, method='linear', fill_value=0).reshape(H,W,3)
    # image2 = np.clip(img_test, 0, 255)  # 确保图像值在 [0, 1] 之间

    # # 将其转换为 0-255 之间的整数值，以符合图片格式的要求
    # image2_uint8 = (image2 * 255).astype(np.uint8)

    # # 保存图片
    # image_pil = Image.fromarray(image2_uint8)
    # image_pil.save("lucid_image_test.png")
    # pts_colors = pts_colors.reshape(H, W, 3)
    
    
    R, T = cam2.R, cam2.T[:, np.newaxis]

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
    
    # qaq = origin_coord - pixel_coord_cam2

    # valid_idx = np.where(np.logical_and.reduce((pixel_coord_cam2[2]>0, 
    #                                             pixel_coord_cam2[0]/pixel_coord_cam2[2]>=0, 
    #                                             pixel_coord_cam2[0]/pixel_coord_cam2[2]<=W-1, 
    #                                             pixel_coord_cam2[1]/pixel_coord_cam2[2]>=0, 
    #                                             pixel_coord_cam2[1]/pixel_coord_cam2[2]<=H-1)))[0]
    # pixel_coord_cam2 = pixel_coord_cam2[:, :,:2]/pixel_coord_cam2[:, :, -1:]

    x, y = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing='xy')
    grid = np.stack((x,y), axis=-1).reshape(-1,2)
    image2 = interp_grid(pixel_coord_cam2[valid_idx], pts_colors[valid_idx], grid, method='linear', fill_value=0).reshape(H,W,3)
    image2 = np.clip(image2, 0, 1)  # 确保图像值在 [0, 1] 之间

    # 将其转换为 0-255 之间的整数值，以符合图片格式的要求
    image2_uint8 = (image2 * 255).astype(np.uint8)

    # 保存图片
    image_pil = Image.fromarray(image2_uint8)
    image_pil.save("lucid_image2.png")
    save_image(img2, "lucid_target_image.png")
    save_image(img1, "lucid_origin_image.png")
    render_depth_or_mask_images("lucid_depth1.png", cam1.depth_image)
    render_depth_or_mask_images("lucid_depth2.png", cam2.depth_image)
    # save_image(image2, "lucid_output_image.png")
    # save_tensor_as_image(image2, "lucid_projected_img1.png")
    
    # new_img2 = compute_epipolar_projection(cam1, cam2, img1, H, W)
    # save_image(new_img2, "output_image.png")
    
    
def prepare_output_and_logger(args):
    print("Training on " + args.source_path)
    
    args.model_path = os.path.join("./output", "blender", os.path.basename(args.source_path))
    
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)

    # All done
    print("\nTraining complete.")
