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
import sys
import cv2
import uuid
import imageio
import wandb
import numpy as np
from argparse import ArgumentParser, Namespace
from pathlib import Path
from random import randint

import torch
from PIL import Image
from tqdm import tqdm

from arguments import ModelParams, OptimizationParams, PipelineParams
from gaussian_renderer import network_gui, render
from scene import GaussianModel, Scene
from utils.loss_utils import l1_loss, ssim
from utils.general_utils import safe_state
from utils.style_utils import FASTLoss, NNFMLoss, KNNFMLoss, GRAMLoss

from preprocess import PreProcess, render_RGBcolor_images, render_depth_or_mask_images


def training(
    dataset,
    opt,
    pipe,
    saving_iterations,
    checkpoint_iterations,
    checkpoint,
    debug_from,
    style_image_dir,
    style_prompt,
    scene_prompt,
    method,
    erode,
    isolate,
    color_transfer,
    stage_one, 
    stage_two,
    style_hyper,
    content_hyper
):
    opt.densify_from_iter = 0
    opt.densification_interval = 50
    opt.opacity_reset_interval = 10000
    if color_transfer:
        opt.densify_until_iter = stage_one
        opt.style_until_iter = stage_one + stage_two
        opt.iterations = stage_one + stage_two + stage_one
    else:
        opt.densify_until_iter = 1
        opt.style_until_iter = stage_two
        opt.iterations = stage_two

    first_iter = 0
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, -1, shuffle=False)
    
    prepare_output_and_logger(dataset, style_image_dir, method)
    
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    bg = torch.rand((3), device="cuda") if opt.random_background else background

    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)
    viewpoint_stack = scene.getTrainCameras().copy()
    
    gaussians.training_setup(opt)
    
    pre = PreProcess(scene, style_image_dir, scene_prompt, style_prompt, 
                     pipe, bg, "cuda", method, 
                     erode, isolate, color_transfer)
    
    if method == "nnfm":
        loss_fn = NNFMLoss(pre, None)
    elif method == "knnfm":
        loss_fn = KNNFMLoss(pre, None)
    elif method == "fast":
        # loss_fn = FASTLoss(pre, [1, 0])
        loss_fn = FASTLoss(pre, None)
    elif method == "gram":
        loss_fn = GRAMLoss(pre, None)
    
    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)
            
    progress_bar = tqdm(range(0, opt.iterations), desc="Training progress")
    first_iter += 1
    

    for iteration in range(1, opt.iterations + 1):
        if network_gui.conn is None:
            network_gui.try_connect()
        while network_gui.conn is not None:
            try:
                net_image_bytes = None
                (
                    custom_cam,
                    do_training,
                    pipe.convert_SHs_python,
                    pipe.compute_cov3D_python,
                    keep_alive,
                    scaling_modifer,
                ) = network_gui.receive()
                if custom_cam is not None:
                    net_image = render(
                        custom_cam, gaussians, pipe, background, scaling_modifer
                    )["render"]
                    net_image_bytes = memoryview(
                        (torch.clamp(net_image, min=0, max=1.0) * 255)
                        .byte()
                        .permute(1, 2, 0)
                        .contiguous()
                        .cpu()
                        .numpy()
                    )
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and (
                    (iteration < int(opt.iterations)) or not keep_alive
                ):
                    break
            except Exception:
                network_gui.conn = None

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        # if iteration % 1000 == 0:
        #     gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        render_pkg = render(viewpoint_cam, gaussians, pipe, bg)

        image, depth, viewspace_point_tensor, visibility_filter, radii = (
            render_pkg["render"],
            render_pkg["depth"],
            render_pkg["viewspace_points"],
            render_pkg["visibility_filter"],
            render_pkg["radii"],
        )
        
        # print("??")
        render_RGBcolor_images("./image.jpg", image)
        render_depth_or_mask_images("./depth.jpg", depth)

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        depth_image = viewpoint_cam.depth_image.cuda()
        scene_mask = viewpoint_cam.scene_mask.cuda()


        h, w = depth_image.shape
        
        if iteration == opt.densify_until_iter:
            
            gaussians.use_filter()
            initial_opacity = gaussians._opacity.clone().detach()
            initial_scaling = gaussians._scaling.clone().detach()
            gaussians._scaling.requires_grad_(False)
            gaussians._xyz.requires_grad_(False)
            gaussians._opacity.requires_grad_(False)
            
            
        if iteration == opt.style_until_iter:     
                  
            if color_transfer: 
                with torch.no_grad():
                    # pre.style_image = pre.original_style_image
                    # pre.style_masks = pre.original_style_masks
                    viewpoint_stack = scene.getTrainCameras()
                    for i, view in enumerate(viewpoint_stack):
                        pkg = render(view, gaussians, pipe, bg)                
                        view.original_image = pkg["render"]
                    pre.gaussian_masks = pre.get_gaussian_masks(pre.scene_weights)
                    pre.color_transfer(pre.gaussian_masks)    
                    viewpoint_stack = None
            
            gaussians._scaling.requires_grad_(False)
            gaussians._xyz.requires_grad_(False)
            gaussians._opacity.requires_grad_(False)
            
            # gaussians._scaling.requires_grad_(True)
        elif iteration < opt.densify_until_iter or iteration > opt.style_until_iter:
            Ll1 = l1_loss(image, gt_image)
            loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
            loss.backward()
        else:
            depth_loss = torch.mean((depth_image - depth) ** 2)
            
            style_loss, content_loss, img_tv_loss = loss_fn(scene_mask, gt_image, image)
            
            loss_delta_opacity = torch.norm(gaussians._opacity - initial_opacity)
            loss_delta_scaling = torch.norm(gaussians._scaling - initial_scaling)
            # loss_delta_xyz = torch.norm(gaussians._xyz - initial_xyz)         
            

            loss = (
                # 10 * style_loss
                style_hyper * style_loss
                # + content_hyper * content_loss
                + 0.02 * img_tv_loss
                + 0.01 * depth_loss
                # + loss_delta_opacity
                # + loss_delta_scaling
                # + 0.05 * loss_delta_xyz
            )
            
            
            # Log and save
            # training_report(
            #     iteration,
            #     style_loss,
            #     content_loss,
            #     img_tv_loss,
            #     loss_delta_opacity,
            #     loss_delta_scaling,
            #     loss,
            #     iter_start.elapsed_time(iter_end),
            # )

            loss.backward()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            loss_for_log = loss.item()
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()
                
            # Densification
            if iteration < opt.densify_until_iter or iteration > opt.style_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)
                
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()
                    

            if iteration == opt.iterations - 1:
                
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration, dataset.model_path)

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)

            if iteration in checkpoint_iterations:
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save(
                    (gaussians.capture(), iteration),
                    scene.model_path + "/chkpnt" + str(iteration) + ".pth",
                )

    pre.render_viewpoint(False, False, False, True, dataset.model_path)
    # os.makedirs("./debug/pre", exist_ok=True)
    # os.makedirs("./debug/aft", exist_ok=True)
    
    # viewpoint_stack = pre.viewpoint_stack
    # for i, view in enumerate(viewpoint_stack):
    #     pkg = render(view, gaussians, pipe, bg)
    #     render_RGBcolor_images(f"./debug/pre/{i}.jpg", pkg["render"])
        
    #     view.original_image = pkg["render"]
        
    # pre.gaussian_masks = pre.get_gaussian_masks(pre.scene_weights)
    # pre.color_transfer(pre.gaussian_masks)
        
    
    # for i, view in enumerate(viewpoint_stack):
    #     pkg = render(view, gaussians, pipe, bg)
    #     render_RGBcolor_images(f"./debug/aft/{i}.jpg", view.original_image)
        
        
    
    

def prepare_output_and_logger(args, style_image_dir, method):
    print("Training on " + args.source_path)
    print(style_image_dir)
    
    style_img_base = os.path.basename(style_image_dir)
    # for i, p in enumerate(style_image):
    #     cur_path, _ = os.path.splitext(os.path.basename(p))
    #     style_img_base += cur_path
    # style_img_base += method
        
    args.model_path = os.path.join("./output/style", os.path.basename(args.source_path) + "/" + style_img_base)
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

@torch.no_grad
def training_report(
    iteration,
    style_loss,
    content_loss,
    img_tv_loss,
    loss_delta_opacity,
    loss_delta_scaling,
    loss,
    elapsed,
):
    wandb.log(
        {
            "style_loss": style_loss,
            "content_loss": content_loss,
            "img_tv_loss": img_tv_loss,
            "loss_delta_opacity": loss_delta_opacity,
            "loss_delta_scaling": loss_delta_scaling,
            "loss": loss,
            "elapsed": elapsed,
        },
        step=iteration,
    )


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--ip", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=6009)
    parser.add_argument("--debug_from", type=int, default=-1)
    parser.add_argument("--detect_anomaly", action="store_true", default=False)
    parser.add_argument(
        "--save_iterations",
        nargs="+",
        type=int,
        default=[500, 1000],
    )
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default=None)
    
    parser.add_argument("--style_image", type=str, default=None)
    parser.add_argument("--style_prompt", type=str)
    parser.add_argument("--scene_prompt", type=str)
    parser.add_argument("--method", type=str, default="fast")
    parser.add_argument("--erode", action="store_true")
    parser.add_argument("--isolate", action="store_true")
    parser.add_argument("--color_transfer", action="store_true")
    parser.add_argument("--stage_one", type=int, default=400)
    parser.add_argument("--stage_two", type=int, default=600)
    parser.add_argument("--style_hyper", type=float, default=2)
    parser.add_argument("--content_hyper", type=float, default=0.005)

    
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    # run = wandb.init(
    #     project="ArtGaussian", notes="wandb first experiment", tags=["baseline"], config=args
    # )

    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(
        lp.extract(args),
        op.extract(args),
        pp.extract(args),
        args.save_iterations,
        args.checkpoint_iterations,
        args.start_checkpoint,
        args.debug_from,
        args.style_image,
        args.style_prompt,
        args.scene_prompt,
        args.method,
        args.erode,
        args.isolate,
        args.color_transfer,
        args.stage_one,
        args.stage_two,
        args.style_hyper,
        args.content_hyper
    )

    # All done
    print("\nTraining complete.")