import sys
sys.path.append("./gs")
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
workspace_dir = os.path.dirname(script_dir)
sys.path.append(workspace_dir)

import torch
import os
from tqdm import tqdm
import imageio
import numpy as np
from gs.scene.cameras import Camera
from gt2gs.style_config import parse_args
from gt2gs.style_trainer import StyleTrainer
from gt2gs.style_utils import render_RGBcolor_images
from scripts.video_utils import c2w2camInfo
from scripts.video_utils import camInfo2c2w
from scripts.video_utils import render_video
from scripts.video_utils import normalize
from scripts.video_utils import viewmatrix

# Please refer to 
# https://github.com/dvlab-research/Ref-NPR/blob/b738f3dd0e21ae5718c46f88d02be48e8bcfcbc1/opt/util/load_llff.py#L337
    
def poses_avg(poses):
    # poses [images, 3, 4] not [images, 3, 5]
    # hwf = poses[0, :3, -1:]

    center = poses[:, :3, 3].mean(0)
    vec2 = normalize(poses[:, :3, 2].sum(0))
    # ic(vec2)
    up = poses[:, :3, 1].sum(0)
    c2w = np.concatenate([viewmatrix(vec2, up, center)], 1)

    return c2w

def render_path_spiral(c2w, up, rads, focal, zrate, rots, N):
    render_poses = []
    rads = np.array(list(rads) + [1.0])
    # hwf = c2w[:,4:5]

    for theta in np.linspace(0.0, 2.0 * np.pi * rots, N + 1)[:-1]:
        c = np.dot(
            c2w[:3, :4],
            np.array([np.cos(theta), -np.sin(theta), -np.sin(theta * zrate), 1.0])
            * rads,
        )
        z = normalize(c - np.dot(c2w[:3, :4], np.array([0, 0, -focal, 1.0])))
        # render_poses.append(np.concatenate([viewmatrix(z, up, c), hwf], 1))
        render_poses.append(viewmatrix(z, up, c))
    render_poses = np.stack(render_poses)
    return render_poses


def generate_spiral_poses(config, poses, bds):
    # For Nerf Coordinate
    poses[:, :3, 1:3] *= -1
    
    c2w = poses_avg(poses)
    up = normalize(poses[:, :3, 1].sum(0))
    close_depth, inf_depth = bds.min() * 0.9, bds.max() * 5.0
    dt = 0.75
    mean_dz = 1.0 / (((1.0 - dt) / close_depth + dt / inf_depth))
    focal = mean_dz
    
    tt = poses[:, :3, 3]
    rads = np.percentile(np.abs(tt), 90, 0)
    c2w_path = c2w
    
    render_c2ws = render_path_spiral(
        c2w_path, up, rads, focal, zrate=0.5, 
        rots=config.num_rotations, N=config.num_frames)
    
    # For 3DGS Coordinate
    render_c2ws[:, :3, 1:3] *= -1
    return render_c2ws


@torch.no_grad
def main():
    # Load Style Trainer
    config = parse_args()
    config.model.model_path = config.style.stylized_model_path
    trainer = StyleTrainer(config)
    
    train_cam_infos = trainer.scene.getTrainCameras()
    train_c2ws = camInfo2c2w(train_cam_infos)
    # Load zfar and znear for spiral camera poses
    bds = np.load(os.path.join(trainer.config.model.source_path, "poses_bounds.npy"))
    bds = bds[:, -2:].transpose([1, 0])
    
    render_c2ws = generate_spiral_poses(trainer.config.video, train_c2ws, bds)
    render_cam_infos = c2w2camInfo(render_c2ws, train_cam_infos[0])
    
    save_path = os.path.join(trainer.config.model.model_path, trainer.config.video.save_frames_path)
    render_video(trainer, save_path, render_cam_infos)
    
    
if __name__ == "__main__":
    main()