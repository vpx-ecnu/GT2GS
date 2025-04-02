from gt2gs.phase.geometry_phase.base_geometry_phase import GeometryPhase
import torch
from gt2gs.style_utils import color_transfer
            

class PostProcessPhase(GeometryPhase):
    
    @torch.no_grad
    def on_phase_start(self):
        
        viewpoint_stack = self.trainer.scene.getTrainCameras()
        for i, view in enumerate(viewpoint_stack):
            pkg = self.trainer.get_render_pkgs(view)
            self.trainer.ctx.scene_images[i] = pkg["render"].detach()
            
        if self.config.style.color_transfer:
            color_transfer(self.trainer.ctx)