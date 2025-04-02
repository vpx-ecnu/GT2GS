import torch
import wandb
from gt2gs.phase.stylize_phase.base_stylize_phase import StylizePhase
from gt2gs.style_utils import *
from gt2gs.style_loss import *


class NNFMPhase(StylizePhase):
    
    def on_iteration(self, iteration: int):
        
        viewpoint_cam = self._get_viewpoint_cam()
        
        with self.trainer.timer as timer:
            
            self.render_pkg = self.trainer.get_render_pkgs(viewpoint_cam)
            render_image = self.render_pkg["render"]
            
            self.render_feat = self.feature_extractor(render_image)
            
            curr_scene_features_mask = self.trainer.ctx.scene_features_mask_list[viewpoint_cam.uid]
            depth_group_num = self.trainer.config.style.depth_group_num
            render_features_list = get_separated_list(self.render_feat, 
                                                      curr_scene_features_mask,
                                                      depth_group_num)
            
            fc, fh, fw = self.render_feat.shape
            target_feat = torch.zeros((fc, fh, fw), device=self.render_feat.device)
            target_matrix = torch.zeros((1, fh, fw), device=self.render_feat.device)
            
            for i in range(depth_group_num):
                mask = (curr_scene_features_mask == i)
                if mask.sum() == 0:
                    continue
                
                style_feat = self.trainer.ctx.style_features_list[i]
                style_matrix = self.trainer.ctx.style_matrix_list[i]
                render_feat = render_features_list[i]
                
                # ic(style_feat.shape, style_matrix.shape, render_feat.shape, (curr_scene_features_mask == i).sum())
                curr_target_feat, curr_target_matrix = nnfm_feat_replace(render_feat, style_feat, style_matrix)
                target_feat[:, mask] = curr_target_feat
                target_matrix[:, mask] = curr_target_matrix
                
            consistent_loss = 0
                
            
            
            # Calculate depth loss
            render_depth = self.render_pkg["depth"]
            curr_depth = self.trainer.ctx.depth_images[viewpoint_cam.uid]
            depth_loss = torch.mean((render_depth - curr_depth) ** 2)
            
            nnfm_loss = cos_distance(target_feat, self.render_feat)
            content_loss = content_loss_fn(render_features_list, 
                                           self.trainer.ctx.scene_features_list[viewpoint_cam.uid])
            
            top2_values, _ = torch.topk(self.trainer.gaussians.get_scaling, k=2, dim=1) 
            shape_loss = (top2_values[:, 0] / top2_values[:, 1]).mean()
            imgtv_loss = get_imgtv_loss(render_image)
            
            
            loss = (
                # Todo: check consistent_loss
                self.trainer.config.style.lambda_consistent_loss * consistent_loss
                + self.trainer.config.style.lambda_nnfm_loss * nnfm_loss
                + self.trainer.config.style.lambda_content_loss * content_loss
                + self.trainer.config.style.lambda_imgtv_loss * imgtv_loss
                + self.trainer.config.style.lambda_depth_loss * depth_loss
                + self.trainer.config.style.lambda_shape_loss * shape_loss
            )
            
            self.update(iteration, loss)
            
            if self.trainer.config.app.need_log:
                wandb.log({
                    "Loss": loss.item(),
                    "Prior Loss": nnfm_loss.item(),
                    "Content Loss": content_loss.item(),
                    "ImgTV Loss": imgtv_loss.item(),
                    "Depth Loss": depth_loss.item(),
                    "Shape Loss": shape_loss.item()
                })
            
        
        return {
            "Points": f"{self.trainer.gaussians._opacity.shape[0]}",
            "Loss": f"{loss.item():.{7}f}"
        }, timer.elapsed_ms
