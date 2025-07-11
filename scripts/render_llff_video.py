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
from gs.utils.video_utils import *
from gs.scene.cameras import Camera
from gt2gs.style_config import parse_args
from gt2gs.style_trainer import StyleTrainer
from gt2gs.style_utils import render_RGBcolor_images
from gs.utils.camera_utils import c2w2camInfo
from gs.utils.camera_utils import camInfo2c2w


def generate_spiral_poses(config, poses, bds):
    # For Nerf Coordinate
    poses[:, :3, 1:3] *= -1
    
    # Please refer to 
    # https://github.com/dvlab-research/Ref-NPR/blob/b738f3dd0e21ae5718c46f88d02be48e8bcfcbc1/opt/util/load_llff.py#L337
    
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


def render_video(trainer, cam_infos):
    
    config = trainer.config
    save_path = os.path.join(config.model.model_path, config.video.save_frames_path)
    os.makedirs(save_path, exist_ok=True)
        
    frames = []
    for idx, cam in enumerate(tqdm(cam_infos, desc="Rendering progress")):
        img = trainer.get_render_pkgs(cam)["render"]
        if config.video.enable_save_frames:
            render_RGBcolor_images(os.path.join(save_path, f"frames/frame_{idx:03d}.jpg"), img)
        frames.append(img)
    # N C H W -> N H W C
    frames = torch.stack(frames).clamp(min=0.0, max=1.0).permute(0, 2, 3, 1).cpu().numpy()
    frames = (frames * 255).astype(np.uint8)
    video_path = os.path.join(save_path, "video.mp4")
    imageio.mimwrite(video_path, frames, fps=30, quality=5)
    print(f'The video is saved in {video_path}.')


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
    
    render_video(trainer, render_cam_infos)
    
    
if __name__ == "__main__":
    main()