import torch
from random import randint
from abc import ABC, abstractmethod
from typing import Dict
from gt2gs.style_utils import render_RGBcolor_images
from icecream import ic 

class TrainingPhase(ABC):
    def __init__(self, trainer, uid, name, start_iter, end_iter, enable_densify=False):
        self.trainer = trainer
        self.config = trainer.config
        self.start_iter = start_iter
        self.end_iter = end_iter
        self.viewpoint_stack = None
        self.name = name
        self.uid = uid
        self.enable_densify = enable_densify
        
    def update(self, iteration, loss):
        
        # ic(self.trainer.gaussians._original_features_dc.sum())
        # ic(self.trainer.gaussians._features_rest.sum())
        loss.backward()
        self._densification(iteration)
        self.trainer.gaussians.optimizer.step()
        self.trainer.gaussians.optimizer.zero_grad(set_to_none=True)

    def on_phase_start(self): ...

    @abstractmethod
    def on_iteration(self, iteration: int) -> Dict[str, torch.Tensor]: ...

    def on_phase_end(self): ...
    
    @torch.no_grad
    def _densification(self, iteration: int): ...
    
    
    def _get_viewpoint_cam(self):
        if not self.viewpoint_stack or len(self.viewpoint_stack) == 0:
            self.viewpoint_stack = self.trainer.scene.getTrainCameras().copy()
        return self.viewpoint_stack.pop(randint(0, len(self.viewpoint_stack) - 1))