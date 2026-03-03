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


import sys
sys.path.append("./gs")
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
workspace_dir = os.path.dirname(script_dir)
sys.path.append(workspace_dir)


from icecream import ic
import torch
from gs.scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gs.gaussian_renderer import render
import torchvision
from gs.utils.general_utils import safe_state
from argparse import ArgumentParser
from gs.arguments import ModelParams, PipelineParams, get_combined_args
from gs.scene import GaussianModel
import cv2
import imageio
import numpy as np
from gs.scene.cameras import Camera
from gt2gs.style_config import parse_args
from gt2gs.style_trainer import StyleTrainer
from scripts.video_utils import c2w2camInfo
from scripts.video_utils import camInfo2c2w
from scripts.video_utils import render_video
from scripts.video_utils import viewmatrix

# Please refer to
# https://github.com/JianlingWANG2021/SGSST/blob/main/make_rendering_videos.py
def integrate_weights_np(w):
    """Compute the cumulative sum of w, assuming all weight vectors sum to 1.

    The output's size on the last dimension is one greater than that of the input,
    because we're computing the integral corresponding to the endpoints of a step
    function, not the integral of the interior/bin values.

    Args:
        w: Tensor, which will be integrated along the last axis. This is assumed to
        sum to 1 along the last axis, and this function will (silently) break if
        that is not the case.

    Returns:
        cw0: Tensor, the integral of w, where cw0[..., 0] = 0 and cw0[..., -1] = 1
    """
    cw = np.minimum(1, np.cumsum(w[..., :-1], axis=-1))
    shape = cw.shape[:-1] + (1,)
    # Ensure that the CDF starts with exactly 0 and ends with exactly 1.
    cw0 = np.concatenate([np.zeros(shape), cw,
                          np.ones(shape)], axis=-1)
    return cw0

def invert_cdf_np(u, t, w_logits):
    """Invert the CDF defined by (t, w) at the points specified by u in [0, 1)."""
    # Compute the PDF and CDF for each weight vector.
    w = np.exp(w_logits) / np.exp(w_logits).sum(axis=-1, keepdims=True)
    cw = integrate_weights_np(w)
    # Interpolate into the inverse CDF.
    interp_fn = np.interp
    t_new = interp_fn(u, cw, t)
    return t_new

def sample_np(rand,
              t,
              w_logits,
              num_samples,
              single_jitter=False,
              deterministic_center=False):
    """
        numpy version of sample()
    """
    eps = np.finfo(np.float32).eps

    # Draw uniform samples.
    if not rand:
        if deterministic_center:
            pad = 1 / (2 * num_samples)
            u = np.linspace(pad, 1. - pad - eps, num_samples)
        else:
            u = np.linspace(0, 1. - eps, num_samples)
        u = np.broadcast_to(u, t.shape[:-1] + (num_samples,))
    else:
        # `u` is in [0, 1) --- it can be zero, but it can never be 1.
        u_max = eps + (1 - eps) / num_samples
        max_jitter = (1 - u_max) / (num_samples - 1) - eps
        d = 1 if single_jitter else num_samples
        u = np.linspace(0, 1 - u_max, num_samples) + \
            np.random.rand(*t.shape[:-1], d) * max_jitter

    return invert_cdf_np(u, t, w_logits)



def focus_point_fn(poses):
    """Calculate nearest point to all focal axes in poses."""
    directions, origins = poses[:, :3, 2:3], poses[:, :3, 3:4]
    m = np.eye(3) - directions * np.transpose(directions, [0, 2, 1])
    mt_m = np.transpose(m, [0, 2, 1]) @ m
    focus_pt = np.linalg.inv(mt_m.mean(0)) @ (mt_m @ origins).mean(0)[:, 0]
    return focus_pt



def pad_poses(p):
    """Pad [..., 3, 4] pose matrices with a homogeneous bottom row [0,0,0,1]."""
    bottom = np.broadcast_to([0, 0, 0, 1.], p[..., :1, :4].shape)
    return np.concatenate([p[..., :3, :4], bottom], axis=-2)


def unpad_poses(p):
    """Remove the homogeneous bottom row from [..., 4, 4] pose matrices."""
    return p[..., :3, :4]

def transform_poses_pca(poses):
    """Transforms poses so principal components lie on XYZ axes.

    Args:
        poses: a (N, 3, 4) array containing the cameras' camera to world transforms.

    Returns:
        A tuple (poses, transform), with the transformed poses and the applied
        camera_to_world transforms.
    """
    t = poses[:, :3, 3]
    t_mean = t.mean(axis=0)
    t = t - t_mean

    eigval, eigvec = np.linalg.eig(t.T @ t)
    # Sort eigenvectors in order of largest to smallest eigenvalue.
    inds = np.argsort(eigval)[::-1]
    eigvec = eigvec[:, inds]
    rot = eigvec.T
    if np.linalg.det(rot) < 0:
        rot = np.diag(np.array([1, 1, -1])) @ rot

    transform = np.concatenate([rot, rot @ -t_mean[:, None]], -1)
    poses_recentered = unpad_poses(transform @ pad_poses(poses))
    transform = np.concatenate([transform, np.eye(4)[3:]], axis=0)

    # Flip coordinate system if z component of y-axis is negative
    if poses_recentered.mean(axis=0)[2, 1] < 0:
        poses_recentered = np.diag(np.array([1, -1, -1])) @ poses_recentered
        transform = np.diag(np.array([1, -1, -1, 1])) @ transform

    # Just make sure it's it in the [-1, 1]^3 cube
    scale_factor = 1. / np.max(np.abs(poses_recentered[:, :3, 3]))
    poses_recentered[:, :3, 3] *= scale_factor
    transform = np.diag(np.array([scale_factor] * 3 + [1])) @ transform
    return poses_recentered, transform

def generate_ellipse_path(config, poses):

    poses[:, :, 1:3] *= -1
    poses, transform = transform_poses_pca(poses)

    # Calculate the focal point for the path (cameras point toward this).
    center = focus_point_fn(poses)
    offset = np.array([center[0] , center[1],  center[2]*0 ])
    # Calculate scaling for ellipse axes based on input camera positions.
    sc = np.percentile(np.abs(poses[:, :3, 3] - offset), 90, axis=0) * 1.2

    # Use ellipse that is symmetric about the focal point in xy.
    low = -sc + offset
    high = sc + offset

    def get_positions(theta):
        # Interpolate between bounds with trig functions to get ellipse in x-y.
        # Optionally also interpolate in z to change camera height along path.
        return np.stack([
            (low[0] + (high - low)[0] * (np.cos(theta) * .5 + .5)),
            (low[1] + (high - low)[1] * (np.sin(theta) * .5 + .5)),
            np.zeros((config.num_frames + 1)),
        ], -1)

    theta = np.linspace(0, 2. * np.pi, config.num_frames + 1, endpoint=True)
    positions = get_positions(theta)

    lengths = np.linalg.norm(positions[1:] - positions[:-1], axis=-1)
    theta = sample_np(None, theta, np.log(lengths), config.num_frames + 1)
    positions = get_positions(theta)

    # Throw away duplicated last position.
    positions = positions[:-1]

    # Set path's up vector to axis closest to average of input pose up vectors.
    avg_up = poses[:, :3, 1].mean(0)
    avg_up = avg_up / np.linalg.norm(avg_up)
    ind_up = np.argmax(np.abs(avg_up))
    up = np.eye(3)[ind_up] * np.sign(avg_up[ind_up])

    render_poses = []
    for p in positions:
        render_pose = viewmatrix(p - center, up, p)
        render_pose = np.linalg.inv(transform) @ render_pose
        render_pose[:3, 1:3] *= -1
        render_poses.append(render_pose)
        
    render_poses = np.stack(render_poses)

    return render_poses

@torch.no_grad
def main():
    
    config = parse_args()
    config.model.model_path = config.style.stylized_model_path
    
    trainer = StyleTrainer(config)
    train_cam_infos = trainer.scene.getTrainCameras()
    train_c2ws = camInfo2c2w(train_cam_infos)
    
    render_c2ws = generate_ellipse_path(trainer.config.video, train_c2ws)
    render_cam_infos = c2w2camInfo(render_c2ws, train_cam_infos[0])
    
    save_path = os.path.join(trainer.config.model.model_path, trainer.config.video.save_frames_path)
    render_video(trainer, save_path, render_cam_infos)
    
    
if __name__ == "__main__":
    main()