import torch 
from gt2gs.phase.base_phase import TrainingPhase
from gt2gs.style_utils import render_RGBcolor_images
from gt2gs.style_utils import render_viewpoint
from torch import nn

class StylizePhase(TrainingPhase):
    
    @torch.no_grad
    def _densification(self, iteration: int):
        if not self.enable_densify:
            return 
        
        gaussians = self.trainer.gaussians
        opt = self.trainer.config.opt
        scene = self.trainer.scene
        
        viewspace_point_tensor = self.render_pkg["viewspace_points"]
        visibility_filter = self.render_pkg["visibility_filter"]
        radii = self.render_pkg["radii"]
        
        gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
        gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

        if (iteration == self.end_iter):
            gaussians.densify_and_prune(opt.style_densification_threshold, 0.005, scene.cameras_extent, 20)
            
    @torch.no_grad
    def on_phase_start(self):
        
        self.feature_extractor = self.trainer.feature_extractor