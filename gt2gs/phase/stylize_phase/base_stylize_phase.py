import torch 
from gt2gs.phase.base_phase import TrainingPhase
from gt2gs.style_utils import render_RGBcolor_images
from gt2gs.style_utils import render_viewpoint
from torch import nn

class StylizePhase(TrainingPhase):
    
    def update(self, iteration, loss):
        super().update(iteration, loss)
        render_RGBcolor_images("./image.jpg", self.render_pkg["render"])
    
    @torch.no_grad
    def _densification(self, iteration: int):
        if not self.enable_densify:
            return 
        
        gaussians = self.trainer.gaussians
        opt = self.trainer.config.opt
        scene = self.trainer.scene
        # dataset = self.trainer.config.model
        
        viewspace_point_tensor = self.render_pkg["viewspace_points"]
        visibility_filter = self.render_pkg["visibility_filter"]
        radii = self.render_pkg["radii"]
        
        gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
        gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)
        if (iteration == self.end_iter):
        
            # ic("densification")
            # threshold = torch.quantile(radii[visibility_filter].float(), 0.10)
            
            # big_mask = torch.logical_and(visibility_filter, radii > threshold)
            # gaussians.split_special_gaussians(big_mask, 10)
            
            # tmp = torch.max(gaussians.get_scaling, dim=1).values
            # threshold = torch.quantile(tmp, 0.80)
            # gaussians.split_special_gaussians(tmp > threshold)
            # self.trainer.gaussians._features_dc = nn.Parameter(self.trainer.gaussians._original_features_dc.clone().requires_grad_(True))
            
            # render_viewpoint(self.trainer, "./debug/densi_bef")
            # self.trainer.gaussians._features_dc = nn.Parameter(self.trainer.gaussians._original_features_dc.clone().requires_grad_(True))
            # render_viewpoint(self.trainer, "./debug/densi_aft")
            # exit(0)
            
            gaussians.densify_and_prune(opt.style_densification_threshold, 0.005, scene.cameras_extent, 20)
            
            # self.trainer.gaussians._features_dc = nn.Parameter(self.trainer.gaussians._original_features_dc.clone().requires_grad_(True))
            # render_viewpoint(self.trainer, "./debug/densi_aft")
            # exit(0)
    @torch.no_grad
    def on_phase_start(self):
        # self.target_feats = {}
        # self.target_matrixs = {}
        # self.projection = {}
        self.feature_extractor = self.trainer.feature_extractor
        # self.original_feats = self.trainer.ctx.original_feats
        # self.style_feat = self.trainer.ctx.style_feat
        # self.style_matrix = self.trainer.ctx.style_matrix
        if self.trainer.config.style.no_grad:
            self.trainer.gaussians._xyz.requires_grad_(False)
            self.trainer.gaussians._rotation.requires_grad_(False)
            self.trainer.gaussians._scaling.requires_grad_(False)
            self.trainer.gaussians._opacity.requires_grad_(False)
        # self._init_projection()
    
    def on_phase_end(self):
        # pass
        if self.trainer.config.style.no_grad:
            self.trainer.gaussians._xyz.requires_grad_(True)
            self.trainer.gaussians._rotation.requires_grad_(True)
            self.trainer.gaussians._scaling.requires_grad_(True)
            self.trainer.gaussians._opacity.requires_grad_(True)
        else:
            pass