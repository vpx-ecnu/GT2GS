import torch
from gt2gs.phase.geometry_phase.base_geometry_phase import GeometryPhase
from gt2gs.style_utils import render_viewpoint
from icecream import ic 

class PreProcessPhase(GeometryPhase):
    

    @torch.no_grad
    def on_phase_start(self):
        pass
        # if self.config.style.color_transfer:
        #     color_transfer(self.trainer.ctx)
            
    @torch.no_grad
    def on_phase_end(self):
        # self.trainer.feature_extractor = FeatureExtractor()
        # self._init_original_feats()
        # self._init_style_feat()
        # _init_depth_images(self.trainer)
        
        # self.trainer.initial_opacity = self.trainer.gaussians._opacity.clone().detach()
        # self.trainer.initial_scaling = self.trainer.gaussians._scaling.clone().detach()
        
        # self.trainer.gaussians.parent_opacity = self.trainer.gaussians._opacity.clone().detach()
        # self.trainer.gaussians.parent_scaling = self.trainer.gaussians._scaling.clone().detach()
        # self.trainer.gaussians.parent_position = self.trainer.gaussians._xyz.clone().detach()
        self.trainer.gaussians._original_features_dc = self.trainer.gaussians._features_dc.clone().detach()
        # render_viewpoint(self.trainer, "./debug/bef")
        
        # self.trainer.gaussians._scaling.requires_grad_(False)
        # self.trainer.gaussians._opacity.requires_grad_(False)
        # self.trainer.gaussians._xyz.requires_grad_(False)
        # self.trainer.gaussians._rotation.requires_grad_(False)
        
        # ic(self.trainer.gaussians._original_features_dc.sum())
        # loss_delta_opacity = torch.norm(self.trainer.gaussians._opacity - self.trainer.gaussians.parent_opacity)
        # loss_delta_scaling = torch.norm(self.trainer.gaussians._scaling - self.trainer.gaussians.parent_opacity)
        # ic(loss_delta_opacity, loss_delta_scaling)
        # exit(0)
        # ic(self.trainer.gaussians.parent_opacity.shape)
        # ic(self.trainer.gaussians.parent_scaling.shape)
        
        # viewpoint_stack = self.trainer.scene.getTrainCameras()
        # self.trainer.ctx.depth_images = []
        
        # for _, view in enumerate(viewpoint_stack):
        #     depth_image = self.trainer.get_render_pkgs(view)["depth"]
        #     self.trainer.ctx.depth_images.append(depth_image.squeeze().detach())
            
        # self.trainer.ctx.depth_images = torch.stack(self.trainer.ctx.depth_images).to(device=self.trainer.device)
 
    # def _init_original_feats(self):
    #     self.trainer.ctx.original_feats = []
        
    #     for i, original_image in enumerate(self.trainer.ctx.scene_images):
    #         self.trainer.ctx.original_feats.append(self.trainer.feature_extractor(original_image))
    
    #     self.trainer.ctx.original_feats = torch.stack(self.trainer.ctx.original_feats)
    
    # def _init_style_feat(self):
    #     pass
        