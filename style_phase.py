# style_phase.py
from abc import ABC, abstractmethod
from enum import Enum, auto
from typing import Dict
import time 
from random import randint
import torch
from style_utils import color_transfer

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
    
    @torch.no_grad
    def on_phase_start(self):
        
        if self.config.style.color_transfer:
            color_transfer(self.trainer.ctx)


    def on_iteration(self, iteration: int) -> Dict[str, torch.Tensor]:
        
        viewpoint_cam = self._get_viewpoint_cam()
        
        with self.trainer.timer as timer:
            
            time.sleep(0.005)
            # self._densify_gaussians(iteration)

        return {"no": 1}, timer.elapsed_ms
    
    def _get_viewpoint_cam(self):
        if not self.viewpoint_stack:
            self.viewpoint_stack = self.trainer.scene.getTrainCameras().copy()
        return self.viewpoint_stack.pop(randint(0, len(self.viewpoint_stack) - 1))

    def _densification(self, iteration: int):
        return

class StylizationPhase(TrainingPhase):
    
    def on_iteration(self, iteration: int) -> Dict[str, torch.Tensor]:
        
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
           