import torch
from typing import Dict
from gs.utils.loss_utils import l1_loss
from gs.utils.loss_utils import ssim
from gt2gs.phase.base_phase import TrainingPhase
from random import randint

class LockParameterPhase(TrainingPhase):
    
    def on_iteration(self, iteration):
        return {}, 0

    def on_phase_start(self):
        self.trainer.gaussians._xyz.requires_grad_(False)
        self.trainer.gaussians._scaling.requires_grad_(False)
        self.trainer.gaussians._opacity.requires_grad_(False)
        self.trainer.gaussians._rotation.requires_grad_(False)