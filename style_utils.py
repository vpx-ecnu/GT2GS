# style_utils.py
import torch 
import numpy as np
import cv2
import os
from gaussian_renderer import render

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
    

    
def color_transfer(ctx):
    original_size = ctx.original_images.size()
    original_pixels = ctx.original_images.reshape(-1, 3)
    style_pixels = ctx.style_image.reshape(-1, 3)
    
    color_transfered_pixels, color_tf = match_colors(original_pixels, style_pixels)
    ctx.original_images = color_transfered_pixels.reshape(*original_size)
        
        
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
    
    
def render_depth_or_mask_images(path, image):
    
    image = image.detach().cpu().numpy().squeeze()
    depth_map_normalized = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
    depth_map_normalized = np.uint8(depth_map_normalized)
    # depth_map_colored = cv2.applyColorMap(depth_map_normalized, cv2.COLORMAP_JET)
    cv2.imwrite(path, depth_map_normalized)
    
def render_RGBcolor_images(path, image):
    
    image = image.detach().permute(1, 2, 0).clamp(min=0.0, max=1.0).cpu().numpy()
    image = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, image)


def read_and_resize_image(image_path, target_height):
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    original_height, original_width = image_rgb.shape[:2]
    scale_ratio = target_height / original_height
    target_width = int(original_width * scale_ratio)
    
    resized_image = cv2.resize(image_rgb, (target_width, target_height), interpolation=cv2.INTER_AREA)
    resized_image = torch.from_numpy(resized_image).permute(2, 0, 1).float() / 255.0
    return resized_image
    

def render_viewpoint(self, if_depth, if_mask, if_original, if_render, path="./debug"):
    
    depth_path = os.path.join(path, "depth/")
    mask_path = os.path.join(path, "mask/")
    original_path = os.path.join(path, "original/")
    render_path = os.path.join(path, "render/")
    
    os.makedirs(depth_path, exist_ok=True)
    os.makedirs(mask_path, exist_ok=True)
    os.makedirs(original_path, exist_ok=True)
    os.makedirs(render_path, exist_ok=True)
    
    for i, view in enumerate(self.viewpoint_stack):
        images_pkgs = render(view, self.scene.gaussians, self.pipe, self.bg)
        
        if if_depth:
            depth_image = images_pkgs["depth"]
            cur_depth_path = os.path.join(depth_path, f"{int(i):04d}.png")
            render_depth_or_mask_images(cur_depth_path, depth_image)
        
        if if_mask:
            mask_image = self.scene_masks[i]
            cur_mask_path = os.path.join(mask_path, f"{int(i):04d}.png")
            render_depth_or_mask_images(cur_mask_path, mask_image)
            
        if if_original:
            original_image = view.original_image
            cur_original_path = os.path.join(original_path, f"{int(i):04d}.png")
            render_RGBcolor_images(cur_original_path, original_image)
            
        if if_render:
            render_image = images_pkgs["render"]
            cur_render_path = os.path.join(render_path, f"{int(i):04d}.png")
            render_RGBcolor_images(cur_render_path, render_image)
            