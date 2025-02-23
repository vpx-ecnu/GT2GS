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