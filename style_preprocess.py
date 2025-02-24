# style_preprocess.py
from dataclasses import dataclass 
import torch 
from typing import List, Optional, Tuple, Dict
from gaussian_renderer import render
from style_utils import read_and_resize_image, render_RGBcolor_images, render_depth_or_mask_images
import os

@dataclass 
class StyleContext:
    
    image_width: int = None
    image_height: int = None
    
    original_images: Optional[torch.Tensor] = None
    depth_images: Optional[torch.Tensor] = None
    style_image: Optional[torch.Tensor] = None
    # scene_mask: Optional[torch.Tensor] = None
    # style_mask: Optional[torch.Tensor] = None

    
        
def _init_scene_images(trainer):
    
    viewpoint_stack = trainer.scene.getTrainCameras()
    trainer.ctx.depth_images = []
    trainer.ctx.original_images = []
    
    # colmap maybe change image's size
    min_h, min_w = 10000, 10000
    for i, view in enumerate(viewpoint_stack):
        min_h = min(min_h, view.image_height)
        min_w = min(min_w, view.image_width)
    trainer.ctx.image_width = min_w
    trainer.ctx.image_height = min_h
    
    for _, view in enumerate(viewpoint_stack):
        depth_image = trainer.get_render_pkgs(view)["depth"]
        trainer.ctx.depth_images.append(depth_image.squeeze().detach())
        trainer.ctx.original_images.append(view.original_image[:, :min_h, :min_w])
        
    trainer.ctx.depth_images = torch.stack(trainer.ctx.depth_images).to(device=trainer.device)
    trainer.ctx.original_images = torch.stack(trainer.ctx.original_images).to(device=trainer.device)
        
    
def _init_style_images(trainer):
    
    trainer.ctx.style_image = read_and_resize_image(
        trainer.config.style.style_image, 
        trainer.config.style.style_image_size,
    ).to(device=trainer.device).contiguous()
        
        
def preprocess(trainer):
    trainer.ctx = StyleContext()
    _init_scene_images(trainer)
    _init_style_images(trainer)
    
    
    
    
    
        
    
        
    