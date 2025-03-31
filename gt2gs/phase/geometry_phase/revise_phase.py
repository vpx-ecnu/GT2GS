import torch
from gt2gs.phase.geometry_phase.base_geometry_phase import GeometryPhase
from gt2gs.style_utils import render_RGBcolor_images
from gt2gs.style_utils import render_viewpoint
from torch import nn
from icecream import ic

class RevisePhase(GeometryPhase):
    
    def update(self, iteration, loss):
        super().update(iteration, loss)
        render_RGBcolor_images("./image.jpg", self.render_pkg["render"])
        
    @torch.no_grad
    def on_phase_start(self):
        
        # render_viewpoint(self.trainer, "./debug/mid")
        
        
        self.tmp_features = self.trainer.gaussians._features_dc.clone().detach()
        
        new_features_dc = self.trainer.gaussians._original_features_dc
        optimizable_tensors = self.trainer.gaussians.replace_tensor_to_optimizer(new_features_dc, "f_dc")
        self.trainer.gaussians._features_dc = optimizable_tensors["f_dc"]
        
        # ic(self.trainer.gaussians._original_features_dc.sum())
        # self.trainer.gaussians._features_dc = nn.Parameter(torch.zeros_like(self.trainer.gaussians._features_dc, device=self.trainer.device).requires_grad_(True))
        # self.trainer.gaussians._features_rest = nn.Parameter(torch.zeros_like(self.trainer.gaussians._features_rest, device=self.trainer.device).requires_grad_(True))
        # render_viewpoint(self.trainer, "./debug/aft")
        # exit(0)
        # self.trainer.gaussians._features_dc.requires_grad_(False)
        # self.trainer.gaussians._scaling.requires_grad_(False)
        # self.trainer.gaussians._opacity.requires_grad_(False)
        # self.feature_extractor = self.trainer.feature_extractor
        # if self.config.style.color_transfer:
        #     color_transfer(self.trainer.ctx)

        # render_ctx(self.trainer.ctx)
        # exit(0)
    
    def on_phase_end(self):
        # self.trainer.gaussians._features_dc.requires_grad_(True)
        # self.trainer.gaussians._features_rest.requires_grad_(True)
        new_features_dc = self.tmp_features.requires_grad_(True)
        optimizable_tensors = self.trainer.gaussians.replace_tensor_to_optimizer(new_features_dc, "f_dc")
        self.trainer.gaussians._features_dc = optimizable_tensors["f_dc"]
        
        # self.trainer.gaussians._scaling.requires_grad_(True)
        # self.trainer.gaussians._opacity.requires_grad_(True)