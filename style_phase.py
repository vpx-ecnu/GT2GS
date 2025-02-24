# style_phase.py
from abc import ABC, abstractmethod
from enum import Enum, auto
from typing import Dict
import time 
from random import randint
import torch
from style_utils import *
from utils.loss_utils import l1_loss, ssim

class TrainingPhaseType(Enum):
    COLOR_TRANSFER_1 = auto()
    STYLIZATION = auto()
    COLOR_TRANSFER_2 = auto()
    

class TrainingPhase(ABC):
    def __init__(self, trainer, start_iter, end_iter):
        self.trainer = trainer
        self.config = trainer.config
        self.start_iter = start_iter
        self.end_iter = end_iter
        self.viewpoint_stack = None

    def on_phase_start(self): ...

    @abstractmethod
    def on_iteration(self, iteration: int) -> Dict[str, torch.Tensor]: ...

    def on_phase_end(self): ...
    
    @abstractmethod
    def _get_viewpoint_cam(self): ...
    
    @abstractmethod
    def _densification(self, iteration: int): ...


class ColorTransferPhase(TrainingPhase):
    
    # @torch.no_grad
    def on_phase_start(self):
        
        if self.config.style.color_transfer:
            color_transfer(self.trainer.ctx)

        # render_ctx(self.trainer.ctx)
        # exit(0)

    def on_iteration(self, iteration: int) -> Dict[str, torch.Tensor]:
        
        viewpoint_cam = self._get_viewpoint_cam()
        # viewpoint_cam = self.trainer.scene.getTrainCameras()[0]
        
        with self.trainer.timer as timer:
            
            self.render_pkg = self.trainer.get_render_pkgs(viewpoint_cam)
            render_image, render_depth = self.render_pkg["render"], self.render_pkg["depth"]
            original_image = self.trainer.ctx.original_images[viewpoint_cam.uid]
            original_depth = self.trainer.ctx.depth_images[viewpoint_cam.uid]
            
            concat_and_save_images("./image.jpg", original_image, render_image, original_depth, render_depth)

            
            Ll1 = l1_loss(render_image, original_image)
            ssim_val = ssim(render_image, original_image)
            
            # print("")
            # print(Ll1, ssim_val, render_image.mean(), original_image.mean(), viewpoint_cam.uid)
            
            loss = (
                (1.0 - self.trainer.config.opt.lambda_dssim) * Ll1 + 
                self.trainer.config.opt.lambda_dssim * (1.0 - ssim_val)
            )
            # exit(0)
            loss.backward()
            self.trainer.gaussians.optimizer.step()
            self.trainer.gaussians.optimizer.zero_grad(set_to_none=True)
            
        self._densification(iteration)
        return {
            "Points": f"{self.trainer.gaussian._opacity.shape[0]}",
            "Loss": f"{loss.item():.{7}f}"
        }, timer.elapsed_ms
    
    def _get_viewpoint_cam(self):
        if not self.viewpoint_stack:
            self.viewpoint_stack = self.trainer.scene.getTrainCameras().copy()
        return self.viewpoint_stack.pop(randint(0, len(self.viewpoint_stack) - 1))

    def _densification(self, iteration: int):
        gaussians = self.trainer.gaussians
        opt = self.trainer.config.opt
        scene = self.trainer.scene
        dataset = self.trainer.config.model
        
        viewspace_point_tensor = self.render_pkg["viewspace_points"]
        visibility_filter = self.render_pkg["visibility_filter"]
        radii = self.render_pkg["radii"]
        
        gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
        gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

        if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
            size_threshold = 20 if iteration > opt.opacity_reset_interval else None
            gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)
        
        if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
            gaussians.reset_opacity()

class StylizationPhase(TrainingPhase):
    
    def on_iteration(self, iteration: int) -> Dict[str, torch.Tensor]:
        exit(0)
        viewpoint_cam = self._get_viewpoint_cam()
        
        with self.trainer.timer as timer:
            
            time.sleep(0.005)
            

        return {"no": 1}, timer.elapsed_ms

    def _get_viewpoint_cam(self):
        if not self.viewpoint_stack:
            self.viewpoint_stack = self.trainer.scene.getTrainCameras()
            self.viewpoint_len = len(self.viewpoint_stack)
            self.viewpoint_idx = -1
        self.viewpoint_idx = (self.viewpoint_idx + 1) % self.viewpoint_len
        return self.viewpoint_stack[self.viewpoint_idx]
    
    
    def _densification(self, iteration: int):
        return
           