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

import cv2
import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from scene import GaussianModel
import numpy as np
from scene.cameras import Camera

def pad_poses(p: np.ndarray) -> np.ndarray:
    """Pad [..., 3, 4] pose matrices with a homogeneous bottom row [0,0,0,1]."""
    bottom = np.broadcast_to([0, 0, 0, 1.], p[..., :1, :4].shape)
    return np.concatenate([p[..., :3, :4], bottom], axis=-2)


def unpad_poses(p: np.ndarray) -> np.ndarray:
    """Remove the homogeneous bottom row from [..., 4, 4] pose matrices."""
    return p[..., :3, :4]


def recenter_poses(poses: np.ndarray):
    """Recenter poses around the origin."""
    cam2world = average_pose(poses)
    transform = np.linalg.inv(pad_poses(cam2world))
    poses = transform @ pad_poses(poses)
    return unpad_poses(poses), transform

def normalize(v):
    """Normalize a vector."""
    return v / np.linalg.norm(v)

def average_pose(poses):
    """
    Calculate the average pose, which is then used to center all poses
    using @center_poses. Its computation is as follows:
    1. Compute the center: the average of pose centers.
    2. Compute the z axis: the normalized average z axis.
    3. Compute axis y': the average y axis.
    4. Compute x' = y' cross product z, then normalize it as the x axis.
    5. Compute the y axis: z cross product x.

    Note that at step 3, we cannot directly use y' as y axis since it's
    not necessarily orthogonal to z axis. We need to pass from x to y.
    Inputs:
        poses: (N_images, 3, 4)
    Outputs:
        pose_avg: (3, 4) the average pose
    """
    # 1. Compute the center
    center = poses[..., 3].mean(0)  # (3)

    # 2. Compute the z axis
    z = normalize(poses[..., 2].mean(0))  # (3)

    # 3. Compute axis y' (no need to normalize as it's not the final output)
    y_ = poses[..., 1].mean(0)  # (3)

    print(np.cross(z, y_))
    # 4. Compute the x axis
    x = normalize(np.cross(z, y_))  # (3)

    # 5. Compute the y axis (as z and x are normalized, y is already of norm 1)
    y = np.cross(x, z)  # (3)

    pose_avg = np.stack([x, y, z, center], 1)  # (3, 4)

    return pose_avg

def viewmatrix(lookdir: np.ndarray, up: np.ndarray,
               position: np.ndarray) -> np.ndarray:
  """Construct lookat view matrix."""
  vec2 = normalize(lookdir)
  vec0 = normalize(np.cross(up, vec2))
  vec1 = normalize(np.cross(vec2, vec0))
  m = np.stack([vec0, vec1, vec2, position], axis=1)
  return m

def render_path_spiral(c2w, up, rads, focal, zdelta, zrate, N_rots=2, N=120):
    render_poses = []
    rads = np.array(list(rads) + [1.])

    for theta in np.linspace(0., 2. * np.pi * N_rots, N + 1)[:-1]:
        c = np.dot(c2w[:3, :4], np.array([np.cos(theta), -np.sin(theta), -np.sin(theta * zrate), 1.]) * rads)
        z = normalize(c - np.dot(c2w[:3, :4], np.array([0, 0, -focal, 1.])))
        render_poses.append(viewmatrix(z, up, c))
    return render_poses

def generate_spiral_path(poses: np.ndarray,
                         bounds: np.ndarray,
                         n_frames: int = 120,
                         n_rots: int = 2,
                         zrate: float = .5) -> np.ndarray:
    """Calculates a forward facing spiral path for rendering."""
        
    NEAR_STRETCH = 0.9  # Push forward near bound for forward facing render path.
    FAR_STRETCH = 5  # Push back far bound for forward facing render path.
    FOCUS_DISTANCE = 0.9  # Relative weighting of near, far bounds for render path.
    # Find a reasonable 'focus depth' for this dataset as a weighted average
    # of conservative near and far bounds in disparity space.
    near_bound = bounds.min() * NEAR_STRETCH
    far_bound = bounds.max() * FAR_STRETCH

    # All cameras will point towards the world space point (0, 0, -focal).
    focal = 60

    # Get radii for spiral path using 90th percentile of camera positions.
    positions = poses[:, :3, 3]
    # print(focal)
    # exit()
    radii = np.percentile(np.abs(positions), 90, 0)
    
    print(positions)
    radii = np.concatenate([radii, [1.]]) 
    
    print(radii)

    # Generate poses for spiral path.
    render_poses = []
    cam2world = average_pose(poses)
    up = poses[:, :3, 1].mean(0)
    for theta in np.linspace(0., 2. * np.pi * n_rots, n_frames, endpoint=False):
        t = radii * [np.cos(theta), -np.sin(theta), -np.sin(theta * zrate), 1.]
        position = cam2world @ t
        lookat = cam2world @ [0, 0, -focal, 1.]
        z_axis = position - lookat
        # print(z_axis, up, position)
        render_poses.append(viewmatrix(z_axis, up, position))
    render_poses = np.stack(render_poses, axis=0)
    return render_poses

def render_set(model_path, name, iteration, views, gaussians, pipeline, background, fps=30):
    render_path = os.path.join(model_path, "render")
    os.makedirs(render_path, exist_ok=True)

    video_path = os.path.join(render_path, 'video.mp4')

    video_writer = None

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        rendering = render(view, gaussians, pipeline, background)["render"]
        image_path = os.path.join(render_path, '{0:04d}.jpg'.format(idx))
        torchvision.utils.save_image(rendering, image_path)
        
        frame = cv2.imread(image_path)

        if video_writer is None:
            height, width, _ = frame.shape
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

        video_writer.write(frame)

        
def create_camera_instances(spiral_poses, fovx, fovy, images, gt_alpha_mask=None):
    cameras = []
    for idx, pose in enumerate(spiral_poses):
        R = pose[:3, :3]
        T = pose[:3, 3]
        camera = Camera(
            colmap_id=idx,
            R=R,
            T=T,
            FoVx=fovx,
            FoVy=fovy,
            image=images[0],
            gt_alpha_mask=gt_alpha_mask,
            image_name=f"image_{idx}.png",
            uid=idx
        )
        cameras.append(camera)
    return cameras

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, skip_path: bool):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
            render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background)

        if not skip_test:
            render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background)
             
        if not skip_path:
            
            all_cameras = scene.getTrainCameras()
            
            # pb = np.load("/data3/lwj/original_data/llff/flower/poses_bounds.npy")
            # print(pb[0])
            # print(all_cameras[0])
            poses = []
            bounds = []
            images = []
            for c in all_cameras:
                pose = np.zeros((3, 4))
                pose[:3, :3] = c.R
                # print(c.R, c.T)
                pose[:3, 3] = c.T
                # print(pose)
                # exit()
                poses.append(pose)
                images.append(c.original_image)
                bd = np.array([c.znear, c.zfar])
                bounds.append(bd)
                
            poses = np.stack(poses)
            bounds = np.stack(bounds)
            # print(bounds)
            
            # scale = 1. / (bounds.min() * .75)
            # poses[:, :3, 3] *= scale
            # bounds *= scale
            
            # print(scale)
            # Recenter poses.
            poses, transform = recenter_poses(poses)
            # Forward-facing spiral render path.
            render_poses = generate_spiral_path(poses, bounds)
            # print(spiral_poses[0])
            # exit()
            fovx = all_cameras[0].FoVx  # Example FoVx in radians
            fovy = all_cameras[0].FoVy  # Example FoVy in radians
            cameras = create_camera_instances(render_poses, fovx, fovy, images, None)
            
            
            render_set(dataset.model_path, "path", scene.loaded_iter, cameras, gaussians, pipeline, background)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--skip_path", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, args.skip_path)