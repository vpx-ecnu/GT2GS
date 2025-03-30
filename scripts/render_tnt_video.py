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

from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation, Slerp

from scipy.special import comb

def bernstein_poly(n, k, t):
    """伯恩斯坦基函数"""
    return comb(n, k) * (t ** k) * ((1 - t) ** (n - k))

def bezier_curve(points, num_frames=60):
    """n阶贝塞尔曲线插值位置"""
    n = len(points) - 1
    t = np.linspace(0, 1, num_frames)
    curve = np.zeros((num_frames, 3))
    for i in range(n + 1):
        curve += np.outer(bernstein_poly(n, i, t), points[i])
    return curve

def squad_interpolation(rotations, num_frames=60):
    """修正后的四元数球面贝塞尔插值（SQUAD）"""
    # 原始关键帧时间戳（确保严格递增）
    original_times = np.linspace(0, 1, len(rotations))
    
    # 生成插值时间戳
    new_times = np.linspace(0, 1, num_frames)
    
    # 将旋转转为四元数
    quats = rotations.as_quat()
    
    # 为SQUAD插值构建控制点和时间戳
    squad_quats = []
    squad_times = []
    
    # 第一个关键帧（重复两次）
    squad_quats.extend([quats[0], quats[0]])
    squad_times.extend([original_times[0], original_times[0] + 1e-6])  # 添加微小增量
    
    # 中间关键帧
    for i in range(1, len(quats)-1):
        q_prev, q, q_next = quats[i-1], quats[i], quats[i+1]
        s = quat_slerp(q_prev, q_next, 0.5)  # 中点插值
        
        # 确保时间戳严格递增
        t_mid = (original_times[i] + original_times[i+1]) / 2
        squad_quats.extend([q, s])
        squad_times.extend([original_times[i], t_mid])
    
    # 最后一个关键帧（重复两次）
    squad_quats.extend([quats[-1], quats[-1]])
    squad_times.extend([original_times[-1] - 1e-6, original_times[-1]])  # 添加微小增量
    
    # 检查时间戳是否严格递增
    assert np.all(np.diff(squad_times) > 0), "Time stamps must be strictly increasing"
    
    # 创建插值器
    squad_rots = Rotation.from_quat(squad_quats)
    return Slerp(squad_times, squad_rots)(new_times)

def simple_slerp_interpolation(rotations, num_frames=60):
    times = np.linspace(0, 1, len(rotations))
    new_times = np.linspace(0, 1, num_frames)
    return Slerp(times, rotations)(new_times)

def quat_slerp(q1, q2, t):
    """四元数球面线性插值辅助函数"""
    q1 = q1 / np.linalg.norm(q1)
    q2 = q2 / np.linalg.norm(q2)
    dot = np.dot(q1, q2)
    
    if dot < 0:
        q2 = -q2
        dot = -dot
    
    theta = np.arccos(np.clip(dot, -1, 1))
    sin_theta = np.sin(theta)
    
    if sin_theta < 1e-10:
        return q1
    
    a = np.sin((1 - t) * theta) / sin_theta
    b = np.sin(t * theta) / sin_theta
    return a * q1 + b * q2

def bezier_camera_path(c2ws_all, num_frames=60):
    """
    输入: 
        c2ws_all: (N, 4, 4) 相机位姿矩阵数组
        num_frames: 输出视频总帧数
    返回:
        render_poses: (num_frames, 4, 4) 插值后的相机位姿
    """
    # 1. 提取控制点（每隔k帧取一个关键帧）
    k = max(1, len(c2ws_all) // 5)  # 控制点数量约5个
    control_points = c2ws_all[::k]
    
    # 2. 位置插值（3D贝塞尔曲线）
    positions = control_points[:, :3, 3]
    interp_positions = bezier_curve(positions, num_frames)
    
    # 3. 旋转插值（SQUAD）
    rotations = Rotation.from_matrix(control_points[:, :3, :3])
    interp_rotations = squad_interpolation(rotations, num_frames)
    
    # 4. 重建相机矩阵
    render_poses = np.zeros((num_frames, 4, 4))
    render_poses[:, :3, :3] = interp_rotations.as_matrix()
    render_poses[:, :3, 3] = interp_positions
    render_poses[:, 3, 3] = 1.0
    
    return render_poses

def interpolate_camera_path(c2ws_all, num_frames=60):
    """
    输入: 
        c2ws_all: (N, 4, 4) 相机位姿矩阵数组
        num_frames: 输出视频总帧数
    返回:
        render_poses: (num_frames, 4, 4) 插值后的相机位姿
    """
    # 1. 提取位置和旋转
    positions = c2ws_all[:, :3, 3]  # (N, 3)
    rotations = Rotation.from_matrix(c2ws_all[:, :3, :3])  # Rotation对象
    
    # 2. 创建插值时间轴
    original_times = np.linspace(0, 1, len(c2ws_all))
    new_times = np.linspace(0, 1, num_frames)
    
    # 3. 位置插值 (3D样条曲线)
    position_splines = [CubicSpline(original_times, positions[:, i]) for i in range(3)]
    interp_positions = np.vstack([spline(new_times) for spline in position_splines]).T
    
    # 4. 旋转插值 (SLERP)
    slerp = Slerp(original_times, rotations)
    interp_rotations = slerp(new_times)
    
    # 5. 重建相机矩阵
    render_poses = np.zeros((num_frames, 4, 4))
    render_poses[:, :3, :3] = interp_rotations.as_matrix()
    render_poses[:, :3, 3] = interp_positions
    render_poses[:, 3, 3] = 1.0  # 齐次坐标
    
    return render_poses

def render_set(model_path, name, iteration, views, gaussians, pipeline, background):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        rendering = render(view, gaussians, pipeline, background)["render"]
        
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".jpg"))

# def render_sets(trainer.config.model : ModelParams, iteration : int, pipeline : PipelineParams):
#     with torch.no_grad():
        
        # if not skip_train:
        #      render_set(trainer.config.model.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background)

        # if not skip_test:
        #      render_set(trainer.config.model.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background)
@torch.no_grad
def main():
    
    config = parse_args()
    config.model.model_path = config.style.stylized_model_path
    
    trainer = StyleTrainer(config)
    

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
    # pose_bounds = np.load(os.path.join('/Datasets/original_data/llff', os.path.basename(trainer.config.model.source_path),'poses_bounds.npy'))
    # depth_ranges = pose_bounds[:, -2:]
    # near_far = [depth_ranges.min(),depth_ranges.max()]
    # cur_near_far = near_far
    
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
    
    # cur_path = get_spiral_render_path(cur_c2ws_all, cur_near_far, rads_scale=1.5, N_views=240)
    # cur_path = interpolate_camera_path(cur_c2ws_all, 240)
    cur_path = cur_c2ws_all
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
    folder = f'{trainer.config.model.model_path}/video/'
    saved_iters = [int(fname.split("_")[-1]) for fname in os.listdir(folder)]
    path = os.path.join(folder, f'ours_{trainer.scene.loaded_iter}/renders')
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
    
    
if __name__ == "__main__":
    
    main()