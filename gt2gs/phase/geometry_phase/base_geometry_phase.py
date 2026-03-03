import torch
from typing import Dict
from gs.utils.loss_utils import l1_loss
from gs.utils.loss_utils import ssim
from gt2gs.phase.base_phase import TrainingPhase
from gt2gs.style_utils import *
from random import randint

class GeometryPhase(TrainingPhase):
    
    def on_iteration(self, iteration: int) -> Dict[str, torch.Tensor]:
        
        viewpoint_cam = self._get_viewpoint_cam()
        
        with self.trainer.timer as timer:
            
            self.render_pkg = self.trainer.get_render_pkgs(viewpoint_cam)
            render_image = self.render_pkg["render"]
            original_image = self.trainer.ctx.scene_images[viewpoint_cam.uid]
            
            Ll1 = l1_loss(render_image, original_image)
            ssim_val = ssim(render_image, original_image)
            
            loss = (
                (1.0 - self.trainer.config.opt.lambda_dssim) * Ll1 + 
                self.trainer.config.opt.lambda_dssim * (1.0 - ssim_val)
            )
            self.update(iteration, loss)

            # concat_and_save_images("./image_post.jpg", render_image, original_image)
              
        return {
            "Points": f"{self.trainer.gaussians._opacity.shape[0]}",
            "Loss": f"{loss.item():.{7}f}"
        }, timer.elapsed_ms
    
        
    @torch.no_grad
    def _densification(self, iteration: int):
        
        if not self.enable_densify:
            return
        
        gaussians = self.trainer.gaussians
        opt = self.trainer.config.opt
        
        # if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == self.start_iter):
        #     gaussians.reset_opacity()
        
        if (
            (iteration - self.start_iter + 1) % opt.densification_interval == 0 and 
            (iteration - self.start_iter) <= (self.end_iter - self.start_iter + 1) // 2
        ):
            tmp = torch.max(gaussians.get_scaling, dim=1).values
            self.threshold = torch.quantile(tmp, 0.95)
            
            
            top2_values, _ = torch.topk(self.trainer.gaussians.get_scaling, k=2, dim=1) 
            ratio = (top2_values[:, 0] / top2_values[:, 1])
            self.threshold_ratio = torch.quantile(ratio, 0.95)
            
            gaussians.split_special_gaussians(torch.logical_or(tmp > self.threshold, ratio > self.threshold_ratio))
            