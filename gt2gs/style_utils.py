# style_utils.py
import torch 
import numpy as np
import cv2
import os
from gs.gaussian_renderer import render
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict
from icecream import ic
from sklearn.cluster import KMeans
import torch.nn.functional as F
    
    
@dataclass
class ProjectContext:
    trans_depth: Optional[torch.Tensor] = None
    trans_pos: Optional[torch.Tensor] = None
    trans_mask: Optional[torch.Tensor] = None


@dataclass 
class StyleContext:
    
    image_width: int = None
    image_height: int = None
    
    scene_images: Optional[torch.Tensor] = None  # [N, C, H, W]
    style_image: Optional[torch.Tensor] = None  # [C, H, W]
    depth_images: Optional[torch.Tensor] = None # [N, C, H, W]
    
    
    # original_feats: Optional[torch.Tensor] = None # [N, C_feature, H//4, W//4]
    # style_feats: Optional[torch.Tensor] = None
    # style_matrix: Optional[torch.Tensor] = None
    
    
    style_features_list: list[torch.Tensor] = None
    style_matrix_list: list[torch.Tensor] = None
    
    scene_masks: Optional[torch.Tensor] = None
    scene_features_list: list[list[torch.Tensor]] = None
    scene_features_mask_list: list[torch.Tensor] = None
    
    # project: Dict[Tuple, ProjectContext] = None
    # scene_mask: Optional[torch.Tensor] = None
    # style_mask: Optional[torch.Tensor] = None

class CUDATimer:
    def __init__(self):
        self.start_event = torch.cuda.Event(enable_timing=True)
        self.end_event = torch.cuda.Event(enable_timing=True)
        self._active = False

    def __enter__(self):
        if self._active:
            raise RuntimeError("Timer is already running")
            
        self.start_event.record()
        self._active = True
        return self

    def __exit__(self, *exc):
        if not self._active:
            return
            
        self.end_event.record()
        self._active = False
        torch.cuda.current_stream().synchronize()

    @property
    def elapsed_ms(self) -> float:
        return self.start_event.elapsed_time(self.end_event)

def cos_distance(a, b):
    a_norm = (a * a).sum(1, keepdims=True).sqrt()
    b_norm = (b * b).sum(1, keepdims=True).sqrt()
    a_tmp = a / (a_norm + 1e-8)
    b_tmp = b / (b_norm + 1e-8)
    cossim = (a_tmp * b_tmp).sum(1)
    cos_d = 1.0 - cossim
    return cos_d.mean()

    
def color_transfer(ctx):
    
    def match_colors(scene_images, style_image):
        sh = scene_images.shape
        image_set = scene_images.view(-1, 3)
        style_img = style_image.view(-1, 3).to(image_set.device)

        mu_c = image_set.mean(0, keepdim=True)
        mu_s = style_img.mean(0, keepdim=True)

        cov_c = torch.matmul((image_set - mu_c).transpose(1, 0), image_set - mu_c).float() / float(image_set.size(0))
        cov_s = torch.matmul((style_img - mu_s).transpose(1, 0), style_img - mu_s).float() / float(style_img.size(0))

        u_c, sig_c, _ = torch.svd(cov_c)
        u_s, sig_s, _ = torch.svd(cov_s)

        u_c_i = u_c.transpose(1, 0)
        u_s_i = u_s.transpose(1, 0)

        scl_c = torch.diag(1.0 / torch.sqrt(torch.clamp(sig_c, 1e-8, 1e8)))
        scl_s = torch.diag(torch.sqrt(torch.clamp(sig_s, 1e-8, 1e8)))

        tmp_mat = u_s @ scl_s @ u_s_i @ u_c @ scl_c @ u_c_i
        tmp_vec = mu_s.view(1, 3) - mu_c.view(1, 3) @ tmp_mat.T

        image_set = image_set @ tmp_mat.T + tmp_vec.view(1, 3)
        image_set = image_set.contiguous().clamp_(0.0, 1.0).view(sh)

        color_tf = torch.eye(4).float().to(tmp_mat.device)
        color_tf[:3, :3] = tmp_mat
        color_tf[:3, 3:4] = tmp_vec.T
        return image_set, color_tf
    
    original_pixels = ctx.scene_images.permute(0, 2, 3, 1)  # [N, C, H, W] -> [N, H, W, C]
    original_size = original_pixels.size() 
    original_pixels = original_pixels.reshape(-1, 3)        # [N, H, W, C] -> [N*H*W, C]
    
    style_pixels = ctx.style_image.permute(1, 2, 0)         # [C, H, W] -> [H, W, C] 
    
    color_transfered_pixels, _ = match_colors(original_pixels, style_pixels)
    
    ctx.scene_images = color_transfered_pixels.reshape(*original_size)
    ctx.scene_images = ctx.scene_images.permute(0, 3, 1, 2) # [N, H, W, C] -> [N, C, H, W]
    
    
    
def render_depth_or_mask_images(path, image):
    
    path_dir = os.path.dirname(path)
    if not os.path.exists(path_dir):
        os.makedirs(path_dir)
    
    image = image.detach().cpu().numpy().squeeze()
    depth_map_normalized = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
    depth_map_normalized = np.uint8(depth_map_normalized)
    depth_map_normalized = cv2.applyColorMap(depth_map_normalized, cv2.COLORMAP_JET)
    cv2.imwrite(path, depth_map_normalized)
    
def render_RGBcolor_images(path, image):
    
    path_dir = os.path.dirname(path)
    if not os.path.exists(path_dir):
        os.makedirs(path_dir)
    
    image = image.detach().permute(1, 2, 0).clamp(min=0.0, max=1.0).cpu().numpy()
    image = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, image)

def convert_RGBcolor_images(image):
    """convert RGB color image and return it as a NumPy array."""
    image = image.detach().permute(1, 2, 0).clamp(min=0.0, max=1.0).cpu().numpy()
    image = (image * 255).astype(np.uint8)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image

def convert_depth_or_mask_images(image):
    """convert depth or mask image and return it as a NumPy array."""
    image = image.detach().cpu().numpy().squeeze()
    depth_map_normalized = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
    depth_map_normalized = np.uint8(depth_map_normalized)
    depth_map_rgb = np.stack([depth_map_normalized] * 3, axis=-1)
    return depth_map_rgb

def concat_and_save_images(output_path, *images, direction='horizontal'):
    
    images_np = [img if isinstance(img, np.ndarray) 
                     else convert_RGBcolor_images(img) 
                              if img.shape[0] == 3 
                              else convert_depth_or_mask_images(img) 
                          for img in images]

    if direction == 'horizontal':
        max_height = max(img.shape[0] for img in images_np)
        concatenated_image = np.hstack([cv2.copyMakeBorder(img, 0, max(max_height - img.shape[0], 0), 0, 0, cv2.BORDER_CONSTANT) for img in images_np])
    elif direction == 'vertical':
        max_width = max(img.shape[1] for img in images_np)
        concatenated_image = np.vstack([cv2.copyMakeBorder(img, 0, 0, 0, max(max_width - img.shape[1], 0), cv2.BORDER_CONSTANT) for img in images_np])
    else:
        raise ValueError("Direction must be either 'horizontal' or 'vertical'.")

    cv2.imwrite(output_path, concatenated_image)

def read_and_resize_image(image_path, target_height):
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    original_height, original_width = image_rgb.shape[:2]
    scale_ratio = target_height / original_height
    target_width = int(original_width * scale_ratio)
    
    resized_image = cv2.resize(image_rgb, (target_width, target_height), interpolation=cv2.INTER_AREA)
    resized_image = torch.from_numpy(resized_image).permute(2, 0, 1).float() / 255.0
    return resized_image


def render_ctx(ctx, path="./debug"):
    
    depth_path = os.path.join(path, "depth/")
    original_path = os.path.join(path, "original/")
    os.makedirs(depth_path, exist_ok=True)
    os.makedirs(original_path, exist_ok=True)
    
    for i in range(ctx.scene_images.shape[0]):
        render_RGBcolor_images(os.path.join(original_path, f"{int(i):04d}.png"), ctx.scene_images[0])
    for i in range(ctx.depth_images.shape[0]):
        render_depth_or_mask_images(os.path.join(depth_path, f"{int(i):04d}.png"), ctx.depth_images[0])


def render_viewpoint(trainer, path=None):
    
    depth_path = os.path.join(path or trainer.config.style.stylized_model_path, "depth/")
    render_path = os.path.join(path or trainer.config.style.stylized_model_path, "render/")
    
    os.makedirs(depth_path, exist_ok=True)
    os.makedirs(render_path, exist_ok=True)
    
    for i, view in enumerate(trainer.scene.getTrainCameras()):
        images_pkgs = trainer.get_render_pkgs(view)
        
        depth_image = images_pkgs["depth"]
        cur_depth_path = os.path.join(depth_path, f"{int(i):04d}.png")
        render_depth_or_mask_images(cur_depth_path, depth_image)
            
        render_image = images_pkgs["render"]
        cur_render_path = os.path.join(render_path, f"{int(i):04d}.png")
        render_RGBcolor_images(cur_render_path, render_image)
            
            
def generate_transformation_matrix(angle, shear_x, shear_y, image_width, image_height):
    """
    Generate the linear transformation matrix for a given rotation angle, shear parameters,
    and image dimensions (to rotate around the image center).
    
    @param angle: Rotation angle in degrees.
    @param shear_x: Shear factor along the x-axis.
    @param shear_y: Shear factor along the y-axis.
    @param image_width: The width of the image.
    @param image_height: The height of the image.
    @return: The combined transformation matrix (3x3).
    """
    # Step 1: Rotation matrix (around the center)
    angle_rad = np.radians(angle)
    rotation_matrix = np.array([
        [np.cos(angle_rad), -np.sin(angle_rad), 0],
        [np.sin(angle_rad), np.cos(angle_rad), 0],
        [0, 0, 1]
    ])
    
    # Step 2: Shear matrix
    shear_matrix = np.array([
        [1, shear_x, 0],
        [shear_y, 1, 0],
        [0, 0, 1]
    ])
    
    # Step 3: Combine rotation and shear matrices
    combined_matrix = np.dot(shear_matrix, rotation_matrix)
    
    # Step 4: Translate image center to origin, apply rotation, then translate back
    center_x = image_width / 2
    center_y = image_height / 2
    
    # Translate to origin, rotate, then translate back
    translation_matrix_to_origin = np.array([
        [1, 0, -center_x],
        [0, 1, -center_y],
        [0, 0, 1]
    ])
    
    translation_matrix_back = np.array([
        [1, 0, center_x],
        [0, 1, center_y],
        [0, 0, 1]
    ])
    
    # Combine all the transformations
    final_matrix = np.dot(translation_matrix_back, np.dot(combined_matrix, translation_matrix_to_origin))
    
    return final_matrix

    
def center_crop(image, crop_size):
    """
    对图像执行中心裁剪。
    
    @param image: 输入图像 (NumPy array)。
    @param crop_size: 裁剪尺寸 (width, height)。
    @return: 裁剪后的图像 (NumPy array)。
    """
    # 获取图像尺寸
    h, w = image.shape[:2]
    
    # 计算裁剪起始点
    start_x = w//2 - crop_size[0]//2
    start_y = h//2 - crop_size[1]//2
    
    # 执行裁剪
    cropped_image = image[start_y:start_y + crop_size[1], start_x:start_x + crop_size[0]]
    
    cropped_image = cv2.resize(cropped_image, (min(h, w), min(h, w)), interpolation=cv2.INTER_LANCZOS4)
    return cropped_image
    
def tensor_img_transformation(tensor_image, transformation_matrix, idx):
    # Convert tensor to NumPy array (H, W, C)
    image_numpy = tensor_image.permute(1, 2, 0).cpu().numpy()
    
    # Get image size
    rows, cols, _ = image_numpy.shape
    # ic(rows, cols)
    # Apply the affine transformation using the top 2x3 part of the transformation matrix
    affine_matrix = transformation_matrix[:2, :]
    transformed_image = cv2.warpAffine(image_numpy, affine_matrix, (cols, rows), flags=cv2.INTER_LANCZOS4)
    # ic(transformed_image.shape)
    
    new_size = int(min(rows, cols) / 1.414)
    
    transformed_image = center_crop(transformed_image, (new_size, new_size))
    # render_RGBcolor_images("./debug/rotate.jpg", torch.from_numpy(transformed_image).permute(2, 0, 1))
    
    # Convert back to Tensor and normalize
    transformed_tensor = torch.from_numpy(transformed_image).permute(2, 0, 1).float().to(tensor_image.device)
    
    return transformed_tensor



class Warper:
    def __init__(self, resolution: tuple = None, device: str = 'gpu0'):
        self.resolution = resolution
        self.device = self.get_device(device)
        return

    def forward_warp(self, frame1: torch.Tensor, mask1: Optional[torch.Tensor], depth1: torch.Tensor,
                     transformation1: torch.Tensor, transformation2: torch.Tensor, intrinsic1: torch.Tensor, 
                     intrinsic2: Optional[torch.Tensor]) -> \
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Given a frame1 and global transformations transformation1 and transformation2, warps frame1 to next view using
        bilinear splatting.
        All arrays should be torch tensors with batch dimension and channel first
        :param frame1: (b, 3, h, w). If frame1 is not in the range [-1, 1], either set is_image=False when calling
                        bilinear_splatting on frame within this function, or modify clipping in bilinear_splatting()
                        method accordingly.
        :param mask1: (b, 1, h, w) - 1 for known, 0 for unknown. Optional
        :param depth1: (b, 1, h, w)
        :param transformation1: (b, 4, 4) extrinsic transformation matrix of first view: [R, t; 0, 1]
        :param transformation2: (b, 4, 4) extrinsic transformation matrix of second view: [R, t; 0, 1]
        :param intrinsic1: (b, 3, 3) camera intrinsic matrix
        :param intrinsic2: (b, 3, 3) camera intrinsic matrix. Optional
        """
        if self.resolution is not None:
            assert frame1.shape[2:4] == self.resolution
        b, c, h, w = frame1.shape
        if mask1 is None:
            mask1 = torch.ones(size=(b, 1, h, w)).to(frame1)
        if intrinsic2 is None:
            intrinsic2 = intrinsic1.clone()

        assert frame1.shape == (b, 3, h, w)
        assert mask1.shape == (b, 1, h, w)
        assert depth1.shape == (b, 1, h, w)
        assert transformation1.shape == (b, 4, 4)
        assert transformation2.shape == (b, 4, 4)
        assert intrinsic1.shape == (b, 3, 3)
        assert intrinsic2.shape == (b, 3, 3)

        frame1 = frame1.to(self.device)
        mask1 = mask1.to(self.device)
        depth1 = depth1.to(self.device)
        transformation1 = transformation1.to(self.device)
        transformation2 = transformation2.to(self.device)
        intrinsic1 = intrinsic1.to(self.device)
        intrinsic2 = intrinsic2.to(self.device)

        # 只用深度图算
        # (b, h, w, 3, 1)
        trans_points1 = self.compute_transformed_points(depth1, transformation1, transformation2, intrinsic1,
                                                        intrinsic2)
        
        # affine_mats = self.compute_feature_affine(depth1, transformation1, transformation2, intrinsic1,
        #                                                 intrinsic2)
        
        trans_coordinates = trans_points1[:, :, :, :2, 0] / trans_points1[:, :, :, 2:3, 0]
        # (h, w, 2) 先前的坐标映射到新坐标系后的坐标
        pos_pre = trans_coordinates.squeeze(0)
        # 因为原来里面是[w, h]，所以要进行变换
        pos_pre = pos_pre[..., [1, 0]]
        
        rows = torch.arange(h, device=self.device)  # (h,)
        cols = torch.arange(w, device=self.device)  # (w,)
        grid_i, grid_j = torch.meshgrid(rows, cols, indexing="ij")
        pos = torch.stack([grid_i, grid_j], dim=-1)  # (h, w, 2)
        
        # 直接拿特征图和图像空间的比例，用图像采样点直接转换为特征图采样点
        # 应该下采样得到一个特征图层面的mask，然后算先验时要区分为mask内的部分和mask外的部分，mask外就是nnfm，mask内就要加上先验条件，包括矩阵方向相似度和特征相似度
        
        
        trans_coordinates = trans_coordinates.permute(0, 3, 1, 2)
        # (b, 1, h, w)
        trans_depth1 = trans_points1[:, :, :, 2, :].permute(0, 3, 1, 2)

        grid = self.create_grid(b, h, w).to(trans_coordinates)
        # (b, 2, h, w)
        flow12 = trans_coordinates - grid

        # TODO: 目前忽略了遮挡关系
        warped_frame2, mask2 = self.bilinear_splatting(frame1, mask1, trans_depth1, flow12, None, is_image=True)
        # warped_depth2 = self.bilinear_splatting(trans_depth1[:, :, None], mask1, trans_depth1, flow12, None,
        #                                         is_image=False)[0][:, :, 0]
        # return warped_frame2, mask2, warped_depth2, flow12
        return warped_frame2, mask2, pos_pre
    
    # def compute_feature_affine(self, depth1, transformation1, transformation2, intrinsic1, intrinsic2):
    #     """
    #     Computes transformed position for each pixel location
    #     """
    #     if self.resolution is not None:
    #         assert depth1.shape[2:4] == self.resolution
    #     b, _, h, w = depth1.shape
    #     if intrinsic2 is None:
    #         intrinsic2 = intrinsic1.clone()
    #     transformation = torch.bmm(transformation2, torch.linalg.inv(transformation1))  # (b, 4, 4)

    #     x1d = torch.arange(0, w)[None]
    #     y1d = torch.arange(0, h)[:, None]
    #     x2d = x1d.repeat([h, 1]).to(depth1)  # (h, w)
    #     y2d = y1d.repeat([1, w]).to(depth1)  # (h, w)
    #     ones_2d = torch.ones(size=(h, w)).to(depth1)  # (h, w)
    #     ones_4d = ones_2d[None, :, :, None, None].repeat([b, 1, 1, 1, 1])  # (b, h, w, 1, 1)
    #     # 原始坐标2d
    #     pos_vectors_homo = torch.stack([x2d, y2d, ones_2d], dim=2)[None, :, :, :, None]  # (1, h, w, 3, 1)

    #     intrinsic1_inv = torch.linalg.inv(intrinsic1)  # (b, 3, 3)
    #     intrinsic1_inv_4d = intrinsic1_inv[:, None, None]  # (b, 1, 1, 3, 3)
    #     intrinsic2_4d = intrinsic2[:, None, None]  # (b, 1, 1, 3, 3)
    #     depth_4d = depth1[:, 0][:, :, :, None, None]  # (b, h, w, 1, 1)
    #     trans_4d = transformation[:, None, None]  # (b, 1, 1, 4, 4)

    #     unnormalized_pos = torch.matmul(intrinsic1_inv_4d, pos_vectors_homo)  # (b, h, w, 3, 1)
    #     # 投到3d世界后
    #     world_points = depth_4d * unnormalized_pos  # (b, h, w, 3, 1)
    #     world_points_homo = torch.cat([world_points, ones_4d], dim=3)  # (b, h, w, 4, 1)
    #     trans_world_homo = torch.matmul(trans_4d, world_points_homo)  # (b, h, w, 4, 1)
    #     trans_world = trans_world_homo[:, :, :, :3]  # (b, h, w, 3, 1)
    #     # 变回新视角的2d
    #     trans_norm_points = torch.matmul(intrinsic2_4d, trans_world)  # (b, h, w, 3, 1)
    #     return trans_norm_points

    def compute_transformed_points(self, depth1: torch.Tensor, transformation1: torch.Tensor, transformation2: torch.Tensor,
                                   intrinsic1: torch.Tensor, intrinsic2: Optional[torch.Tensor]):
        """
        Computes transformed position for each pixel location
        """
        if self.resolution is not None:
            assert depth1.shape[2:4] == self.resolution
        b, _, h, w = depth1.shape
        if intrinsic2 is None:
            intrinsic2 = intrinsic1.clone()
        transformation = torch.bmm(transformation2, torch.linalg.inv(transformation1))  # (b, 4, 4)

        x1d = torch.arange(0, w)[None]
        y1d = torch.arange(0, h)[:, None]
        x2d = x1d.repeat([h, 1]).to(depth1)  # (h, w)
        y2d = y1d.repeat([1, w]).to(depth1)  # (h, w)
        ones_2d = torch.ones(size=(h, w)).to(depth1)  # (h, w)
        ones_4d = ones_2d[None, :, :, None, None].repeat([b, 1, 1, 1, 1])  # (b, h, w, 1, 1)
        # 原始坐标2d
        pos_vectors_homo = torch.stack([x2d, y2d, ones_2d], dim=2)[None, :, :, :, None]  # (1, h, w, 3, 1)

        intrinsic1_inv = torch.linalg.inv(intrinsic1)  # (b, 3, 3)
        intrinsic1_inv_4d = intrinsic1_inv[:, None, None]  # (b, 1, 1, 3, 3)
        intrinsic2_4d = intrinsic2[:, None, None]  # (b, 1, 1, 3, 3)
        depth_4d = depth1[:, 0][:, :, :, None, None]  # (b, h, w, 1, 1)
        trans_4d = transformation[:, None, None]  # (b, 1, 1, 4, 4)

        unnormalized_pos = torch.matmul(intrinsic1_inv_4d, pos_vectors_homo)  # (b, h, w, 3, 1)
        # 投到3d世界后
        world_points = depth_4d * unnormalized_pos  # (b, h, w, 3, 1)
        world_points_homo = torch.cat([world_points, ones_4d], dim=3)  # (b, h, w, 4, 1)
        trans_world_homo = torch.matmul(trans_4d, world_points_homo)  # (b, h, w, 4, 1)
        trans_world = trans_world_homo[:, :, :, :3]  # (b, h, w, 3, 1)
        # 变回新视角的2d
        trans_norm_points = torch.matmul(intrinsic2_4d, trans_world)  # (b, h, w, 3, 1)
        return trans_norm_points

    def bilinear_splatting(self, frame1: torch.Tensor, mask1: Optional[torch.Tensor], depth1: torch.Tensor,
                           flow12: torch.Tensor, flow12_mask: Optional[torch.Tensor], is_image: bool = False) -> \
            Tuple[torch.Tensor, torch.Tensor]:
        """
        Bilinear splatting
        :param frame1: (b,c,h,w)
        :param mask1: (b,1,h,w): 1 for known, 0 for unknown. Optional
        :param depth1: (b,1,h,w)
        :param flow12: (b,2,h,w)
        :param flow12_mask: (b,1,h,w): 1 for valid flow, 0 for invalid flow. Optional
        :param is_image: if true, output will be clipped to (-1,1) range
        :return: warped_frame2: (b,c,h,w)
                 mask2: (b,1,h,w): 1 for known and 0 for unknown
        """
        if self.resolution is not None:
            assert frame1.shape[2:4] == self.resolution
        b, c, h, w = frame1.shape
        if mask1 is None:
            mask1 = torch.ones(size=(b, 1, h, w)).to(frame1)
        if flow12_mask is None:
            flow12_mask = torch.ones(size=(b, 1, h, w)).to(flow12)
        grid = self.create_grid(b, h, w).to(frame1)
        trans_pos = flow12 + grid

        trans_pos_offset = trans_pos + 1
        trans_pos_floor = torch.floor(trans_pos_offset).long()
        trans_pos_ceil = torch.ceil(trans_pos_offset).long()
        trans_pos_offset = torch.stack([
            torch.clamp(trans_pos_offset[:, 0], min=0, max=w + 1),
            torch.clamp(trans_pos_offset[:, 1], min=0, max=h + 1)], dim=1)
        trans_pos_floor = torch.stack([
            torch.clamp(trans_pos_floor[:, 0], min=0, max=w + 1),
            torch.clamp(trans_pos_floor[:, 1], min=0, max=h + 1)], dim=1)
        trans_pos_ceil = torch.stack([
            torch.clamp(trans_pos_ceil[:, 0], min=0, max=w + 1),
            torch.clamp(trans_pos_ceil[:, 1], min=0, max=h + 1)], dim=1)

        prox_weight_nw = (1 - (trans_pos_offset[:, 1:2] - trans_pos_floor[:, 1:2])) * \
                         (1 - (trans_pos_offset[:, 0:1] - trans_pos_floor[:, 0:1]))
        prox_weight_sw = (1 - (trans_pos_ceil[:, 1:2] - trans_pos_offset[:, 1:2])) * \
                         (1 - (trans_pos_offset[:, 0:1] - trans_pos_floor[:, 0:1]))
        prox_weight_ne = (1 - (trans_pos_offset[:, 1:2] - trans_pos_floor[:, 1:2])) * \
                         (1 - (trans_pos_ceil[:, 0:1] - trans_pos_offset[:, 0:1]))
        prox_weight_se = (1 - (trans_pos_ceil[:, 1:2] - trans_pos_offset[:, 1:2])) * \
                         (1 - (trans_pos_ceil[:, 0:1] - trans_pos_offset[:, 0:1]))

        sat_depth1 = torch.clamp(depth1, min=0, max=1000)
        log_depth1 = torch.log(1 + sat_depth1)
        depth_weights = torch.exp(log_depth1 / log_depth1.max() * 50)

        weight_nw = torch.moveaxis(prox_weight_nw * mask1 * flow12_mask / depth_weights, [0, 1, 2, 3], [0, 3, 1, 2])
        weight_sw = torch.moveaxis(prox_weight_sw * mask1 * flow12_mask / depth_weights, [0, 1, 2, 3], [0, 3, 1, 2])
        weight_ne = torch.moveaxis(prox_weight_ne * mask1 * flow12_mask / depth_weights, [0, 1, 2, 3], [0, 3, 1, 2])
        weight_se = torch.moveaxis(prox_weight_se * mask1 * flow12_mask / depth_weights, [0, 1, 2, 3], [0, 3, 1, 2])

        warped_frame = torch.zeros(size=(b, h + 2, w + 2, c), dtype=torch.float32).to(frame1)
        warped_weights = torch.zeros(size=(b, h + 2, w + 2, 1), dtype=torch.float32).to(frame1)

        frame1_cl = torch.moveaxis(frame1, [0, 1, 2, 3], [0, 3, 1, 2])
        batch_indices = torch.arange(b)[:, None, None].to(frame1.device)
        warped_frame.index_put_((batch_indices, trans_pos_floor[:, 1], trans_pos_floor[:, 0]),
                                frame1_cl * weight_nw, accumulate=True)
        warped_frame.index_put_((batch_indices, trans_pos_ceil[:, 1], trans_pos_floor[:, 0]),
                                frame1_cl * weight_sw, accumulate=True)
        warped_frame.index_put_((batch_indices, trans_pos_floor[:, 1], trans_pos_ceil[:, 0]),
                                frame1_cl * weight_ne, accumulate=True)
        warped_frame.index_put_((batch_indices, trans_pos_ceil[:, 1], trans_pos_ceil[:, 0]),
                                frame1_cl * weight_se, accumulate=True)

        warped_weights.index_put_((batch_indices, trans_pos_floor[:, 1], trans_pos_floor[:, 0]),
                                  weight_nw, accumulate=True)
        warped_weights.index_put_((batch_indices, trans_pos_ceil[:, 1], trans_pos_floor[:, 0]),
                                  weight_sw, accumulate=True)
        warped_weights.index_put_((batch_indices, trans_pos_floor[:, 1], trans_pos_ceil[:, 0]),
                                  weight_ne, accumulate=True)
        warped_weights.index_put_((batch_indices, trans_pos_ceil[:, 1], trans_pos_ceil[:, 0]),
                                  weight_se, accumulate=True)

        warped_frame_cf = torch.moveaxis(warped_frame, [0, 1, 2, 3], [0, 2, 3, 1])
        warped_weights_cf = torch.moveaxis(warped_weights, [0, 1, 2, 3], [0, 2, 3, 1])
        cropped_warped_frame = warped_frame_cf[:, :, 1:-1, 1:-1]
        cropped_weights = warped_weights_cf[:, :, 1:-1, 1:-1]

        mask = cropped_weights > 0
        zero_value = -1 if is_image else 0
        zero_tensor = torch.tensor(zero_value, dtype=frame1.dtype, device=frame1.device)
        warped_frame2 = torch.where(mask, cropped_warped_frame / cropped_weights, zero_tensor)
        mask2 = mask.to(frame1)

        if is_image:
            assert warped_frame2.min() >= -1.1  # Allow for rounding errors
            assert warped_frame2.max() <= 1.1
            warped_frame2 = torch.clamp(warped_frame2, min=-1, max=1)
        return warped_frame2, mask2

    def bilinear_interpolation(self, frame2: torch.Tensor, mask2: Optional[torch.Tensor], flow12: torch.Tensor,
                               flow12_mask: Optional[torch.Tensor], is_image: bool = False) -> \
            Tuple[torch.Tensor, torch.Tensor]:
        """
        Bilinear interpolation
        :param frame2: (b, c, h, w)
        :param mask2: (b, 1, h, w): 1 for known, 0 for unknown. Optional
        :param flow12: (b, 2, h, w)
        :param flow12_mask: (b, 1, h, w): 1 for valid flow, 0 for invalid flow. Optional
        :param is_image: if true, output will be clipped to (-1,1) range
        :return: warped_frame1: (b, c, h, w)
                 mask1: (b, 1, h, w): 1 for known and 0 for unknown
        """
        if self.resolution is not None:
            assert frame2.shape[2:4] == self.resolution
        b, c, h, w = frame2.shape
        if mask2 is None:
            mask2 = torch.ones(size=(b, 1, h, w)).to(frame2)
        if flow12_mask is None:
            flow12_mask = torch.ones(size=(b, 1, h, w)).to(flow12)
        grid = self.create_grid(b, h, w).to(frame2)
        trans_pos = flow12 + grid

        trans_pos_offset = trans_pos + 1
        trans_pos_floor = torch.floor(trans_pos_offset).long()
        trans_pos_ceil = torch.ceil(trans_pos_offset).long()
        trans_pos_offset = torch.stack([
            torch.clamp(trans_pos_offset[:, 0], min=0, max=w + 1),
            torch.clamp(trans_pos_offset[:, 1], min=0, max=h + 1)], dim=1)
        trans_pos_floor = torch.stack([
            torch.clamp(trans_pos_floor[:, 0], min=0, max=w + 1),
            torch.clamp(trans_pos_floor[:, 1], min=0, max=h + 1)], dim=1)
        trans_pos_ceil = torch.stack([
            torch.clamp(trans_pos_ceil[:, 0], min=0, max=w + 1),
            torch.clamp(trans_pos_ceil[:, 1], min=0, max=h + 1)], dim=1)

        prox_weight_nw = (1 - (trans_pos_offset[:, 1:2] - trans_pos_floor[:, 1:2])) * \
                         (1 - (trans_pos_offset[:, 0:1] - trans_pos_floor[:, 0:1]))
        prox_weight_sw = (1 - (trans_pos_ceil[:, 1:2] - trans_pos_offset[:, 1:2])) * \
                         (1 - (trans_pos_offset[:, 0:1] - trans_pos_floor[:, 0:1]))
        prox_weight_ne = (1 - (trans_pos_offset[:, 1:2] - trans_pos_floor[:, 1:2])) * \
                         (1 - (trans_pos_ceil[:, 0:1] - trans_pos_offset[:, 0:1]))
        prox_weight_se = (1 - (trans_pos_ceil[:, 1:2] - trans_pos_offset[:, 1:2])) * \
                         (1 - (trans_pos_ceil[:, 0:1] - trans_pos_offset[:, 0:1]))

        weight_nw = torch.moveaxis(prox_weight_nw * flow12_mask, [0, 1, 2, 3], [0, 3, 1, 2])
        weight_sw = torch.moveaxis(prox_weight_sw * flow12_mask, [0, 1, 2, 3], [0, 3, 1, 2])
        weight_ne = torch.moveaxis(prox_weight_ne * flow12_mask, [0, 1, 2, 3], [0, 3, 1, 2])
        weight_se = torch.moveaxis(prox_weight_se * flow12_mask, [0, 1, 2, 3], [0, 3, 1, 2])

        frame2_offset = F.pad(frame2, [1, 1, 1, 1])
        mask2_offset = F.pad(mask2, [1, 1, 1, 1])
        bi = torch.arange(b)[:, None, None]

        f2_nw = frame2_offset[bi, :, trans_pos_floor[:, 1], trans_pos_floor[:, 0]]
        f2_sw = frame2_offset[bi, :, trans_pos_ceil[:, 1], trans_pos_floor[:, 0]]
        f2_ne = frame2_offset[bi, :, trans_pos_floor[:, 1], trans_pos_ceil[:, 0]]
        f2_se = frame2_offset[bi, :, trans_pos_ceil[:, 1], trans_pos_ceil[:, 0]]

        m2_nw = mask2_offset[bi, :, trans_pos_floor[:, 1], trans_pos_floor[:, 0]]
        m2_sw = mask2_offset[bi, :, trans_pos_ceil[:, 1], trans_pos_floor[:, 0]]
        m2_ne = mask2_offset[bi, :, trans_pos_floor[:, 1], trans_pos_ceil[:, 0]]
        m2_se = mask2_offset[bi, :, trans_pos_ceil[:, 1], trans_pos_ceil[:, 0]]

        nr = weight_nw * f2_nw * m2_nw + weight_sw * f2_sw * m2_sw + \
             weight_ne * f2_ne * m2_ne + weight_se * f2_se * m2_se
        dr = weight_nw * m2_nw + weight_sw * m2_sw + weight_ne * m2_ne + weight_se * m2_se

        zero_value = -1 if is_image else 0
        zero_tensor = torch.tensor(zero_value, dtype=nr.dtype, device=nr.device)
        warped_frame1 = torch.where(dr > 0, nr / dr, zero_tensor)
        mask1 = (dr > 0).to(frame2)

        # Convert to channel first
        warped_frame1 = torch.moveaxis(warped_frame1, [0, 1, 2, 3], [0, 2, 3, 1])
        mask1 = torch.moveaxis(mask1, [0, 1, 2, 3], [0, 2, 3, 1])

        if is_image:
            assert warped_frame1.min() >= -1.1  # Allow for rounding errors
            assert warped_frame1.max() <= 1.1
            warped_frame1 = torch.clamp(warped_frame1, min=-1, max=1)
        return warped_frame1, mask1

    @staticmethod
    def create_grid(b, h, w):
        x_1d = torch.arange(0, w)[None]
        y_1d = torch.arange(0, h)[:, None]
        x_2d = x_1d.repeat([h, 1])
        y_2d = y_1d.repeat([1, w])
        grid = torch.stack([x_2d, y_2d], dim=0)
        batch_grid = grid[None].repeat([b, 1, 1, 1])
        return batch_grid

    # @staticmethod
    # def read_image(path: Path) -> torch.Tensor:
    #     image = skimage.io.imread(path.as_posix())
    #     return image

    # @staticmethod
    # def read_depth(path: Path) -> torch.Tensor:
    #     if path.suffix == '.png':
    #         depth = skimage.io.imread(path.as_posix())
    #     elif path.suffix == '.npy':
    #         depth = numpy.load(path.as_posix())
    #     elif path.suffix == '.npz':
    #         with numpy.load(path.as_posix()) as depth_data:
    #             depth = depth_data['depth']
    #     elif path.suffix == '.exr':
    #         exr_file = OpenEXR.InputFile(path.as_posix())
    #         raw_bytes = exr_file.channel('B', Imath.PixelType(Imath.PixelType.FLOAT))
    #         depth_vector = numpy.frombuffer(raw_bytes, dtype=numpy.float32)
    #         height = exr_file.header()['displayWindow'].max.y + 1 - exr_file.header()['displayWindow'].min.y
    #         width = exr_file.header()['displayWindow'].max.x + 1 - exr_file.header()['displayWindow'].min.x
    #         depth = numpy.reshape(depth_vector, (height, width))
    #     else:
    #         raise RuntimeError(f'Unknown depth format: {path.suffix}')
    #     return depth

    @staticmethod
    def camera_intrinsic_transform(capture_width=1920, capture_height=1080, patch_start_point: tuple = (0, 0)):
        start_y, start_x = patch_start_point
        camera_intrinsics = np.eye(4)
        camera_intrinsics[0, 0] = 2100
        camera_intrinsics[0, 2] = capture_width / 2.0 - start_x
        camera_intrinsics[1, 1] = 2100
        camera_intrinsics[1, 2] = capture_height / 2.0 - start_y
        return camera_intrinsics

    @staticmethod
    def get_device(device: str):
        """
        Returns torch device object
        :param device: cpu/gpu0/gpu1
        :return:
        """
        if device == 'cpu':
            device = torch.device('cpu')
        elif device.startswith('gpu') and torch.cuda.is_available():
            gpu_num = int(device[3:])
            device = torch.device(f'cuda:{gpu_num}')
        else:
            device = torch.device('cpu')
        return device
    
    
def normalize_depth_to_uint8(depth_tensor):
    
    min_val = depth_tensor.min()
    max_val = depth_tensor.max()
    
    normalized = (depth_tensor - min_val) / (max_val - min_val + 1e-6)
    normalized = torch.clamp(normalized, 0, 1)
    
    uint8_depth = (normalized * 255).to(torch.uint8)
    
    return uint8_depth

def labels_downscale(labels, new_dim):
    """
    Downscales the labels to a new dimension.

    @param labels: Tensor of labels. Shape: [H, W]
    @param new_dim: Tuple of new dimensions (NH, NW)
    @return: Downscaled labels
    """
    H, W = labels.shape
    NH, NW = new_dim
    r_indices = torch.linspace(0, H-1, NH).long()
    c_indices = torch.linspace(0, W-1, NW).long()
    return labels[r_indices[:, None], c_indices]

def get_separated_list(pixels, mask, num_classes):
    separated_list = []
    for i in range(num_classes):
        separated_list.append(pixels[:, mask == i])
    return separated_list

@torch.no_grad
def get_enhanced_style_features(trainer, style_image):
    
    enhanced_style_features = []
    style_matrix = []
    
    _, style_img_width, style_img_height = style_image.shape
    # ic(style_img_width, style_img_height)
    
    for i in range(0, 360):
        
        Hx = Hy = 0
        
        theta = i * 1.0
        
        # Rotate Image with center crop and get initial features
        M = generate_transformation_matrix(theta, Hx, Hy, style_img_width, style_img_height) # M: [3, 3]
        new_image = tensor_img_transformation(style_image, M, i)
        img_feats = trainer.feature_extractor(new_image, False)
        
        
        ######################################################################
        # 不聚类
        if trainer.config.style.gta_type == "default":
            c, h, w = img_feats.shape
            img_feats = img_feats.view(c, -1)  # [C, H, W] -> [C, H*W]
            _, num_clusters = img_feats.shape
        
        ######################################################################
        # 聚类
        if trainer.config.style.gta_type == "kmeans":
            # 处理为k-means可用形式
            c, h, w = img_feats.shape
            img_feats_flat = img_feats.permute(1, 2, 0).reshape(-1, c)  # [C, H, W] -> [H*W, C]
            img_feats_np = img_feats_flat.cpu().numpy()
            
            # k-means
            # TODO: 超参调整
            num_clusters = 40
            kmeans = KMeans(n_clusters=num_clusters, random_state=42, max_iter=300)
            kmeans.fit(img_feats_np)
            
            # 获取每个像素对应的聚类标签
            cluster_labels = kmeans.labels_  # [h * w]

            # 获取聚类中心
            cluster_centers = kmeans.cluster_centers_  # [num_clusters, c]

            # 将每个像素替换为其所属聚类的中心向量
            discretized_feats_flat = torch.tensor(cluster_centers, device=img_feats.device)

            # 将特征还原回原始形状 [c, -1]
            img_feats = discretized_feats_flat.view(-1, c).permute(1, 0) # [H*W, C] -> [C, H*W]
        
        ######################################################################
        # 裁剪
        # TODO: 后面可以手动控制变量范围
        
        if trainer.config.style.gta_type == "clip":
            c, h, w = img_feats.shape
            target_h = 8
            target_w = 8
            
            start_h = (h - target_h) // 2
            start_w = (w - target_w) // 2
            
            cropped_tensor = img_feats[:, start_h:start_h + target_h, start_w:start_w + target_w]
            # view必须张量在内存中连续，所以用reshape
            img_feats = cropped_tensor.reshape(c, -1)
            _, num_clusters = img_feats.shape
        
        # 将新特征组加入总特征集合（当前未考虑深度分组）
        
        enhanced_style_features.append(img_feats) # [C, num_clusters]
        
        matrix_list = torch.full((1, num_clusters), theta, device=trainer.device) # [1, num_clusters]
        style_matrix.append(matrix_list) 
        

    enhanced_style_features = torch.cat(enhanced_style_features, dim=1) # [C, num_clusters * 360]
    style_matrix = torch.cat(style_matrix, dim=1) # [1, num_clusters * 360]
    
    return enhanced_style_features, style_matrix


@torch.no_grad
def get_original_style_features(trainer, style_image):
    style_features = trainer.feature_extractor(style_image, False)
    c, h, w = style_features.shape
    style_matrix = torch.zeros((1, h * w), device=style_features.device)
    style_features = style_features.reshape(c, -1)
    return style_features, style_matrix


def compute_rotation_angles(A, B, fh, fw):
    """
    计算特征图中每个位置的旋转角度（并行化版本）。
    
    参数：
        A (torch.Tensor): 形状为 [h, w, 2] 的张量，表示原始坐标 [X, Y]
        B (torch.Tensor): 形状为 [h, w, 2] 的张量，表示变换后的坐标 [X, Y]
        fh (int): 特征图的高度
        fw (int): 特征图的宽度
    
    返回：
        C (torch.Tensor): 形状为 [1, fh, fw] 的张量，表示每个特征图像素的旋转角度
    """
    # 获取原图像尺寸
    h, w, _ = A.shape
    
    # 计算池化步幅
    pool_size_h = h // fh
    pool_size_w = w // fw
    
    # 确保尺寸匹配
    # assert h % fh == 0 and w % fw == 0, "特征图尺寸必须能整除原图像尺寸"
    
    # 生成所有 (x, y) 位置的网格
    x_indices = torch.arange(fh, device=A.device)
    y_indices = torch.arange(fw, device=A.device)
    x_grid, y_grid = torch.meshgrid(x_indices, y_indices, indexing='ij')
    
    # 计算四个角点的索引
    top_left_x = x_grid * pool_size_h
    top_left_y = y_grid * pool_size_w
    top_right_x = x_grid * pool_size_h
    top_right_y = (y_grid + 1) * pool_size_w - 1
    bottom_left_x = (x_grid + 1) * pool_size_h - 1
    bottom_left_y = y_grid * pool_size_w
    bottom_right_x = (x_grid + 1) * pool_size_h - 1
    bottom_right_y = (y_grid + 1) * pool_size_w - 1
    
    # 提取所有位置的角点，形状为 (fh, fw, 4, 2)
    A_S = torch.stack([
        A[top_left_x, top_left_y],
        A[top_right_x, top_right_y],
        A[bottom_left_x, bottom_left_y],
        A[bottom_right_x, bottom_right_y]
    ], dim=2).float()
    
    B_S = torch.stack([
        B[top_left_x, top_left_y],
        B[top_right_x, top_right_y],
        B[bottom_left_x, bottom_left_y],
        B[bottom_right_x, bottom_right_y]
    ], dim=2).float()
    
    # 批量中心化
    mu_a = A_S.mean(dim=2, keepdim=True)  # 形状 (fh, fw, 1, 2)
    mu_b = B_S.mean(dim=2, keepdim=True)  # 形状 (fh, fw, 1, 2)
    A_centered = A_S - mu_a  # 形状 (fh, fw, 4, 2)
    B_centered = B_S - mu_b  # 形状 (fh, fw, 4, 2)
    
    # 批量构造 H 矩阵
    H = torch.einsum('fhij,fhjk->fhik', B_centered.permute(0, 1, 3, 2), A_centered)
    
    # 批量 SVD 分解
    U, S, Vh = torch.svd(H)  # U, Vh: (fh, fw, 2, 2), S: (fh, fw, 2)
    
    # 计算旋转矩阵 R = U @ Vh^T
    R = torch.einsum('fhij,fhjk->fhik', U, Vh.permute(0, 1, 3, 2))  # (fh, fw, 2, 2)
    
    # 检查行列式并调整（确保 R 是旋转矩阵）
    det_R = R[:, :, 0, 0] * R[:, :, 1, 1] - R[:, :, 0, 1] * R[:, :, 1, 0]  # (fh, fw)
    mask = det_R < 0
    if mask.any():
        U[mask, :, -1] = -U[mask, :, -1]  # 调整 U 的最后一列
        R[mask] = torch.bmm(U[mask], Vh[mask].permute(0, 2, 1))
    
    # 批量提取角度 theta
    theta = torch.atan2(R[:, :, 1, 0], R[:, :, 0, 0])  # 形状 (fh, fw)
    
    # 赋值给 C
    C = theta.unsqueeze(0)  # 形状 (1, fh, fw)
    
    return C