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
from gs.utils.video_utils import *
from gs.scene.cameras import Camera
from gt2gs.style_config import parse_args
from gt2gs.style_trainer import StyleTrainer


def render_set(model_path, name, iteration, views, gaussians, pipeline, background):
    render_path = os.path.join(model_path, name)

    makedirs(render_path, exist_ok=True)

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        rendering = render(view, gaussians, pipeline, background)["render"]
        
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".jpg"))

# def render_sets(trainer.config.model : ModelParams, iteration : int, pipeline : PipelineParams):
#     with torch.no_grad():
        
        # if not skip_train:
        #      render_set(trainer.config.model.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background)

        # if not skip_test:
        #      render_set(trainer.config.model.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background)
def render_video(trainer):


    cam_infos = trainer.scene.getTrainCameras()
    cam_infos_render_vd = []
    cur_c2ws_all = []
    for train_cam in cam_infos:
        R, T = train_cam.R, train_cam.T
        w2c = np.eye(4)
        w2c[:3,:3] = R
        # TODO：维度压缩
        w2c[:3,3] = T.squeeze()
        c2w = np.linalg.inv(w2c)
        cur_c2ws_all.append(c2w)
    cur_c2ws_all = np.stack(cur_c2ws_all)
    # pose_bounds = np.load(os.path.join('/data3/lwj/original_data/llff', os.path.basename(trainer.config.model.model_path),'poses_bounds.npy')) # c2w, -u, r, -t
    pose_bounds = np.load(os.path.join('/Datasets/original_data/llff', os.path.basename(trainer.config.model.source_path),'poses_bounds.npy'))
    depth_ranges = pose_bounds[:, -2:]
    near_far = [depth_ranges.min(),depth_ranges.max()]
    cur_near_far = near_far
    
    # # 提取所有相机的位置 (N, 3)
    # camera_positions = cur_c2ws_all[:, :3, 3]

    # # 计算场景中心 (均值)
    # scene_center = np.mean(camera_positions, axis=0)

    # # 计算相机到场景中心的最大/最小距离
    # distances = np.linalg.norm(camera_positions - scene_center, axis=1)
    # min_dist = np.min(distances)
    # max_dist = np.max(distances)

    # # 启发式设置 near 和 far
    # near = max(0.1, min_dist * 0.8)  # 避免 near=0
    # far = max_dist * 1.2              # 扩大范围保证覆盖
    
    # cur_near_far[0] = near
    # cur_near_far[1] = far
    
    cur_path = get_spiral_render_path(cur_c2ws_all, cur_near_far, rads_scale=1.5, N_views=240)
    # cur_path = interpolate_camera_path(cur_c2ws_all, 240)
    # cur_path = bezier_camera_path(cur_c2ws_all, 240)
    # convert back to extrinsics tensor
    cur_w2cs = np.linalg.inv(cur_path)[:, :3].astype(np.float32)
    for idx, cur_w2c in enumerate(cur_w2cs):
        R = cur_w2c[:,:3]
        T = cur_w2c[:,3]
        cam_info = Camera(
            colmap_id=idx,
            R=R,
            T=T,
            FoVx=cam_infos[0].FoVx,
            FoVy=cam_infos[0].FoVy,
            image=cam_infos[0].original_image,
            gt_alpha_mask=None,
            image_name=f"image_{idx}.png",
            uid=idx
        )
        cam_infos_render_vd.append(cam_info)



    render_cam_infos = cam_infos_render_vd


    render_set(trainer.config.model.model_path, "video", trainer.scene.loaded_iter, render_cam_infos, trainer.gaussians, trainer.config.pipe, trainer.background)
    

    # generate video
    path = f'{trainer.config.model.model_path}/video/'
    scene_name = os.path.basename(trainer.config.model.model_path)
    filelist = os.listdir(path)
    filelist = sorted(filelist)
    imgs = []
    for item in filelist:
        if item.endswith('.jpg'):
            item = os.path.join(path, item)
            img = cv2.imread(item)[:,:,::-1]
            imgs.append(img)
    # save_dir = trainer.config.model.model_path
    # os.makedirs(save_dir,exist_ok=True)
    save_path = os.path.join(trainer.config.model.model_path,f'{scene_name}.mp4')
    imageio.mimwrite(save_path, np.stack(imgs), fps=60, quality=5)
    print(f'The video is saved in {save_path}.')

    # render_viewpoint(trainer)
    
@torch.no_grad
def main():
    
    config = parse_args()
    config.model.model_path = config.style.stylized_model_path
    
    trainer = StyleTrainer(config)
    render_video(trainer)
    
    
if __name__ == "__main__":
    
    main()