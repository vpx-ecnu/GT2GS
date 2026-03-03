import torch
from gt2gs.phase.geometry_phase.base_geometry_phase import GeometryPhase
from gt2gs.style_utils import render_viewpoint
from icecream import ic 

class PreProcessPhase(GeometryPhase):
    
    @torch.no_grad
    def on_phase_start(self):
        self.trainer.gaussians._original_features_dc = self.trainer.gaussians._features_dc.clone().detach()