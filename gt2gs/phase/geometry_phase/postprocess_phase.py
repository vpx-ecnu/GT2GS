from gt2gs.phase.geometry_phase.base_geometry_phase import GeometryPhase
import torch
from gt2gs.style_utils import color_transfer
            

class PostProcessPhase(GeometryPhase):
    
    @torch.no_grad
    def on_phase_start(self):
        
        # self.viewpoint_stack = self.trainer.scene.getTrainCameras().copy()
        # self.viewpoint_len = len(self.viewpoint_stack)
        # for i in range(0, self.viewpoint_len):
        #     curr_cam = self.viewpoint_stack[i]
        #     self.render_pkg = self.trainer.get_render_pkgs(curr_cam)
        #     render_img = self.render_pkg["render"]
        #     self.trainer.ctx.scene_images[curr_cam.uid] = render_img
        viewpoint_stack = self.trainer.scene.getTrainCameras()
        for i, view in enumerate(viewpoint_stack):
            pkg = self.trainer.get_render_pkgs(view)
            self.trainer.ctx.scene_images[i] = pkg["render"].detach()
            
        if self.config.style.color_transfer:
            color_transfer(self.trainer.ctx)