import os
from tqdm import tqdm
from gt2gs.style_utils import render_RGBcolor_images
import torch
import imageio
import numpy as np
from gs.utils.camera_utils import getWorld2View2
from gs.utils.camera_utils import Camera


def render_video(trainer, save_path, cam_infos):
    
    config = trainer.config
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

def normalize(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def viewmatrix(z, up, pos):
    vec2 = normalize(z)
    vec1_avg = up
    vec0 = normalize(np.cross(vec1_avg, vec2))
    vec1 = normalize(np.cross(vec2, vec0))
    m = np.eye(4)
    m[:3] = np.stack([vec0, vec1, vec2, pos], 1)
    return m

def camInfo2c2w(camInfos):    
    c2ws = []
    for cam in camInfos:
        w2c = getWorld2View2(cam.R, cam.T.squeeze())
        c2w = np.linalg.inv(w2c)
        c2ws.append(c2w)
    c2ws = np.stack(c2ws)
    return c2ws
    
def c2w2camInfo(c2ws, base_info):
    cam_infos = []
    for idx, c2w in enumerate(c2ws):
        w2c = np.linalg.inv(c2w)
        
        cam_infos.append(Camera(
            colmap_id=idx,
            R=w2c[:3, :3].T,
            T=w2c[:3, 3],
            FoVx=base_info.FoVx,
            FoVy=base_info.FoVy,
            image=base_info.original_image,
            gt_alpha_mask=None,
            image_name=f"image_{idx}.png",
            uid=idx
        ))
    return cam_infos