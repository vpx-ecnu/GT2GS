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

from scene.cameras import Camera
import numpy as np
from utils.general_utils import PILtoTorch
from utils.graphics_utils import fov2focal
from utils.graphics_utils import getWorld2View2

WARNED = False

def loadCam(args, id, cam_info, resolution_scale):
    orig_w, orig_h = cam_info.image.size

    if args.resolution in [1, 2, 4, 8]:
        resolution = round(orig_w/(resolution_scale * args.resolution)), round(orig_h/(resolution_scale * args.resolution))
    else:  # should be a type that converts to float
        if args.resolution == -1:
            if orig_w > 1600:
                global WARNED
                if not WARNED:
                    print("[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.\n "
                        "If this is not desired, please explicitly specify '--resolution/-r' as 1")
                    WARNED = True
                global_down = orig_w / 1600
            else:
                global_down = 1
        else:
            global_down = orig_w / args.resolution

        scale = float(global_down) * float(resolution_scale)
        resolution = (int(orig_w / scale), int(orig_h / scale))

    resized_image_rgb = PILtoTorch(cam_info.image, resolution)

    gt_image = resized_image_rgb[:3, ...]
    loaded_mask = None

    if resized_image_rgb.shape[1] == 4:
        loaded_mask = resized_image_rgb[3:4, ...]

    return Camera(colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T, 
                  FoVx=cam_info.FovX, FoVy=cam_info.FovY, 
                  image=gt_image, gt_alpha_mask=loaded_mask,
                  image_name=cam_info.image_name, uid=id, data_device=args.data_device)

def cameraList_from_camInfos(cam_infos, resolution_scale, args):
    camera_list = []

    for id, c in enumerate(cam_infos):
        camera_list.append(loadCam(args, id, c, resolution_scale))

    return camera_list

def camera_to_JSON(id, camera : Camera):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = camera.R.transpose()
    Rt[:3, 3] = camera.T
    Rt[3, 3] = 1.0

    W2C = np.linalg.inv(Rt)
    pos = W2C[:3, 3]
    rot = W2C[:3, :3]
    serializable_array_2d = [x.tolist() for x in rot]
    camera_entry = {
        'id' : id,
        'img_name' : camera.image_name,
        'width' : camera.width,
        'height' : camera.height,
        'position': pos.tolist(),
        'rotation': serializable_array_2d,
        'fy' : fov2focal(camera.FovY, camera.height),
        'fx' : fov2focal(camera.FovX, camera.width)
    }
    return camera_entry



#--------------------------------------------------------
import json
import math

import torch
import torch.nn.functional as F
from PIL import Image
from kornia.core import Tensor, concatenate


def compute_epipolar_projection(cam1, cam2, img1, current_H=64, current_W=64):
    """
    计算 img1 中的像素值通过极线对齐变换到 cam2 中的位置，并生成 cam2 对应的图像 img2。

    Args:
        cam1: 第一个相机对象，包含相机的投影矩阵等信息。
        cam2: 第二个相机对象，包含相机的投影矩阵等信息。
        img1: 第一个相机拍摄的图像，形状为 (C, H, W)。
        current_H: 图像高度，默认为 64。
        current_W: 图像宽度，默认为 64。

    Returns:
        img2: 变换后的图像，形状为 (C, H, W)。
    """
    # 计算基础矩阵
    F_matrix = get_fundamental_matrix_with_H(cam1, cam2, current_H, current_W)

    # 生成 img1 的像素坐标网格
    x = torch.arange(current_W)
    y = torch.arange(current_H)
    x, y = torch.meshgrid(x, y, indexing='xy')
    x = x.reshape(-1)
    y = y.reshape(-1)

    
    # 将像素坐标转为齐次坐标
    heto_cam1 = torch.stack([x, y, torch.ones_like(x)], dim=1).type(F_matrix.dtype).cuda()  # (N, 3)

    print(F_matrix.dtype)
    print(heto_cam1.dtype)
    
    # 通过基础矩阵计算极线
    lines = (F_matrix @ heto_cam1.T).T  # (N, 3)

    # 计算每个点在 cam2 中的对应坐标
    x2 = -lines[:, 2] / (lines[:, 0] + 1e-8)  # 避免除以 0
    y2 = -lines[:, 2] / (lines[:, 1] + 1e-8)
    coords_cam2 = torch.stack([x2, y2], dim=1)  # (N, 2)

    # 将齐次坐标转为图像网格坐标
    coords_cam2 = coords_cam2.view(current_H, current_W, 2)  # (H, W, 2)

    # 对 img1 进行变换，生成 img2
    coords_cam2_normalized = torch.stack([
        coords_cam2[..., 0] / (current_W - 1) * 2 - 1,  # x -> [-1, 1]
        coords_cam2[..., 1] / (current_H - 1) * 2 - 1   # y -> [-1, 1]
    ], dim=-1)  # (H, W, 2)

    img1 = img1.unsqueeze(0)  # 添加批量维度，(1, C, H, W)
    img2 = F.grid_sample(img1, coords_cam2_normalized.unsqueeze(0), align_corners=True)  # (1, C, H, W)

    return img2.squeeze(0)



def load_json_to_cameras(json_path, image_dir, args=None):
    """
    从 JSON 文件加载相机信息，并返回 Camera 对象列表。
    """
    with open(json_path, 'r') as f:
        camera_data = json.load(f)

    cameras = []
    for cam in camera_data:
        
        img_path = f"{image_dir}/{cam['img_name']}.JPG"

        orig_w, orig_h = cam["width"], cam["height"]
        # resolution_scale = args.resolution_scale if hasattr(args, 'resolution_scale') else 1.0
        # if args.resolution in [1, 2, 4, 8]:
        #     resolution = (round(orig_w / (resolution_scale * args.resolution)),
        #                   round(orig_h / (resolution_scale * args.resolution)))
        # else:
        #     resolution = (orig_w, orig_h) 

        resolution = (orig_w, orig_h)
        pil_image = Image.open(img_path)
        gt_image = PILtoTorch(pil_image, resolution)

        camera = Camera(
            colmap_id=cam['id'],
            R=np.array(cam['rotation'], dtype=np.float32),
            T=np.array(cam['position'], dtype=np.float32),
            FoVx=cam['fx'],
            FoVy=cam['fy'],
            image=gt_image,
            gt_alpha_mask=None,  # 假设没有 Alpha 通道
            image_name=cam['img_name'],
            uid=cam['id']
        )

        cameras.append(camera)

    return cameras

@torch.no_grad()
def fundamental_from_projections(P1: Tensor, P2: Tensor) -> Tensor:
    r"""Get the Fundamental matrix from Projection matrices.

    Args:
        P1: The projection matrix from first camera with shape :math:`(*, 3, 4)`.
        P2: The projection matrix from second camera with shape :math:`(*, 3, 4)`.

    Returns:
         The fundamental matrix with shape :math:`(*, 3, 3)`.
    """
    if not (len(P1.shape) >= 2 and P1.shape[-2:] == (3, 4)):
        raise AssertionError(P1.shape)
    if not (len(P2.shape) >= 2 and P2.shape[-2:] == (3, 4)):
        raise AssertionError(P2.shape)
    if P1.shape[:-2] != P2.shape[:-2]:
        raise AssertionError

    def vstack(x: Tensor, y: Tensor) -> Tensor:
        return concatenate([x, y], dim=-2)

    X1 = P1[..., 1:, :]
    X2 = vstack(P1[..., 2:3, :], P1[..., 0:1, :])
    X3 = P1[..., :2, :]

    Y1 = P2[..., 1:, :]
    Y2 = vstack(P2[..., 2:3, :], P2[..., 0:1, :])
    Y3 = P2[..., :2, :]

    X1Y1, X2Y1, X3Y1 = vstack(X1, Y1), vstack(X2, Y1), vstack(X3, Y1)
    X1Y2, X2Y2, X3Y2 = vstack(X1, Y2), vstack(X2, Y2), vstack(X3, Y2)
    X1Y3, X2Y3, X3Y3 = vstack(X1, Y3), vstack(X2, Y3), vstack(X3, Y3)

    F_vec = torch.cat(
        [
            X1Y1.det().reshape(-1, 1),
            X2Y1.det().reshape(-1, 1),
            X3Y1.det().reshape(-1, 1),
            X1Y2.det().reshape(-1, 1),
            X2Y2.det().reshape(-1, 1),
            X3Y2.det().reshape(-1, 1),
            X1Y3.det().reshape(-1, 1),
            X2Y3.det().reshape(-1, 1),
            X3Y3.det().reshape(-1, 1),
        ],
        dim=1,
    )

    return F_vec.view(*P1.shape[:-2], 3, 3)

def get_fundamental_matrix_with_H(cam1, cam2, current_H, current_W):
    NDC_2_pixel = torch.tensor([[current_W / 2, 0, current_W / 2], [0, current_H / 2, current_H / 2], [0, 0, 1]]).cuda()
    # NDC_2_pixel_inversed = torch.inverse(NDC_2_pixel)
    NDC_2_pixel = NDC_2_pixel.float()
    cam_1_transformation = cam1.full_proj_transform[:, [0,1,3]].T.float()
    cam_2_transformation = cam2.full_proj_transform[:, [0,1,3]].T.float()
    cam_1_pixel = NDC_2_pixel@cam_1_transformation
    cam_2_pixel = NDC_2_pixel@cam_2_transformation

    # print(NDC_2_pixel.dtype, cam_1_transformation.dtype, cam_2_transformation.dtype, cam_1_pixel.dtype, cam_2_pixel.dtype)

    cam_1_pixel = cam_1_pixel.float()
    cam_2_pixel = cam_2_pixel.float()
    # print("cam_1", cam_1_pixel.dtype, cam_1_pixel.shape)
    # print("cam_2", cam_2_pixel.dtype, cam_2_pixel.shape)
    # print(NDC_2_pixel@cam_1_transformation, NDC_2_pixel@cam_2_transformation)
    return fundamental_from_projections(cam_1_pixel, cam_2_pixel)
    # return fundamental_from_projections(NDC_2_pixel.float()@cam1.full_proj_transform[:, [0,1,3]].T.float(), NDC_2_pixel.float()@cam2.full_proj_transform[:, [0,1,3]].T.float()).half()

def point_to_line_dist(points, lines):
    """
    Calculate the distance from points to lines in 2D.
    points: Nx3
    lines: Mx3

    return distance: NxM
    """
    numerator = torch.abs(lines @ points.T)
    denominator = torch.linalg.norm(lines[:,:2], dim=1, keepdim=True)
    return numerator / denominator

def compute_epipolar_constrains(cam1, cam2, current_H=64, current_W=64):
    '''
    返回满足极线约束的像素的mask
    '''
    n_frames = 1
    sequence_length = current_W * current_H
    fundamental_matrix_1 = []
    
    fundamental_matrix_1.append(get_fundamental_matrix_with_H(cam1, cam2, current_H, current_W))
    fundamental_matrix_1 = torch.stack(fundamental_matrix_1, dim=0)

    x = torch.arange(current_W)
    y = torch.arange(current_H)
    x, y = torch.meshgrid(x, y, indexing='xy')
    x = x.reshape(-1)
    y = y.reshape(-1)
    heto_cam2 = torch.stack([x, y, torch.ones(size=(len(x),))], dim=1).view(-1, 3).cuda()
    heto_cam1 = torch.stack([x, y, torch.ones(size=(len(x),))], dim=1).view(-1, 3).cuda()
    # epipolar_line: n_frames X seq_len,  3
    line1 = (heto_cam2.unsqueeze(0).repeat(n_frames, 1, 1) @ fundamental_matrix_1.cuda()).view(-1, 3)
    
    distance1 = point_to_line_dist(heto_cam1, line1)

    
    idx1_epipolar = distance1 > 1 # sequence_length x sequence_lengths

    return idx1_epipolar

def get_intrinsic_matrix(camera):

    FoVx = torch.tensor(camera.FoVx, dtype=torch.float32, device='cuda')
    FoVy = torch.tensor(camera.FoVy, dtype=torch.float32, device='cuda')

    f_x = camera.image_width / (2 * torch.tan(FoVx / 2))
    f_y = camera.image_height / (2 * torch.tan(FoVy / 2))

    c_x = camera.image_width / 2
    c_y = camera.image_height / 2

    intrinsic_matrix = torch.tensor([[f_x,0,c_x],
                     [0,f_y,c_y],
                     [0,0,1]],dtype=torch.float32).cuda()

    return intrinsic_matrix

def get_homography_matrix(cam1,cam2):
    '''
    Homography induced by a pure rotation as an approximation.
    Suppose the translations between cameras are small 
    compared to the distances of the scene objects from the cameras, 
    and the cameras' focal lengths are small enough
    '''

    K1 = get_intrinsic_matrix(cam1)
    K2 = get_intrinsic_matrix(cam2)

    R1 = torch.tensor(cam1.R, dtype=torch.float32, device='cuda')  
    R2 = torch.tensor(cam2.R, dtype=torch.float32, device='cuda')

    K1_inverse = torch.linalg.inv(K1)

    R_2_1 = torch.matmul(R2,R1.T)

    H = torch.matmul(K2,torch.matmul(R_2_1,K1_inverse))
    return H

def sort_cameras_by_angle(cameras):
    center = torch.mean(torch.stack([torch.tensor(cam.T, dtype=torch.float32) for cam in cameras]), dim=0)

    def polar_angle(camera):
        relative_position = torch.tensor(camera.T, dtype=torch.float32) - center
        dx, dy = relative_position[0].item(), relative_position[1].item()
        return math.atan2(dy, dx)

    cameras.sort(key=polar_angle)

    # # 使相邻相机之间的视角差异较大
    # half = len(cameras) // 2
    # sorted_cameras = cameras[:half] + cameras[half:][::-1]

    return cameras


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