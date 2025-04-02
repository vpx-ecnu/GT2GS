import torch
from gt2gs.phase.geometry_phase.base_geometry_phase import GeometryPhase
from gt2gs.style_utils import render_RGBcolor_images
from gt2gs.style_utils import render_viewpoint
from torch import nn
from icecream import ic
from gs.utils.loss_utils import l1_loss
from gs.utils.loss_utils import ssim
import wandb

class RevisePhase(GeometryPhase):
    
    def swap_features(self, new_features):
        original_features = self.trainer.gaussians._features_dc.clone().detach()
        optimizable_tensors = self.trainer.gaussians.replace_tensor_to_optimizer(new_features, "f_dc")
        self.trainer.gaussians._features_dc = optimizable_tensors["f_dc"]
        return original_features
    
    @torch.no_grad
    def on_phase_start(self):
        
        self.stylized_features = self.swap_features(self.trainer.gaussians._original_features_dc)
        
        # self.stylized_features = self.trainer.gaussians._features_dc.clone().detach()
        
        # new_features_dc = self.trainer.gaussians._original_features_dc
        # optimizable_tensors = self.trainer.gaussians.replace_tensor_to_optimizer(new_features_dc, "f_dc")
        # self.trainer.gaussians._features_dc = optimizable_tensors["f_dc"]
    
    def on_phase_end(self):
        
        self.trainer.gaussians._original_features_dc = self.swap_features(self.stylized_features)
        
        # self.trainer.gaussians._original_features_dc = self.trainer.gaussians._features_dc.clone().detach()
        
        # new_features_dc = self.stylized_features.requires_grad_(True)
        # optimizable_tensors = self.trainer.gaussians.replace_tensor_to_optimizer(new_features_dc, "f_dc")
        # self.trainer.gaussians._features_dc = optimizable_tensors["f_dc"]
    
    def update(self, iteration, loss):
        super().update(iteration, loss)
        render_RGBcolor_images("./image.jpg", self.render_pkg["render"])
        
        
    def on_iteration(self, iteration: int):
        
        viewpoint_cam = self._get_viewpoint_cam()
        
        with self.trainer.timer as timer:
            
            self.render_pkg = self.trainer.get_render_pkgs(viewpoint_cam)
            render_image = self.render_pkg["render"]
            original_image = viewpoint_cam.original_image
            
            Ll1 = l1_loss(render_image, original_image)
            ssim_val = ssim(render_image, original_image)
            
            loss = (
                (1.0 - self.trainer.config.opt.lambda_dssim) * Ll1 + 
                self.trainer.config.opt.lambda_dssim * (1.0 - ssim_val)
            )
            self.update(iteration, loss)
            
            if self.trainer.config.app.need_log:
                wandb.log({
                    "Loss": loss.item(),
                    "Ll1 Loss": Ll1.item(),
                    "ssim_val Loss": ssim_val.item(),
                })
              
        return {
            "Points": f"{self.trainer.gaussians._opacity.shape[0]}",
            "Loss": f"{loss.item():.{7}f}"
        }, timer.elapsed_ms