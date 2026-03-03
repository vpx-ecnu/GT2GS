import torch
import wandb
from gt2gs.phase.stylize_phase.base_stylize_phase import StylizePhase
from gt2gs.style_utils import *
from gt2gs.style_loss import *


class PriorPhase(StylizePhase):
    
    @torch.no_grad
    def on_phase_start(self):
        super().on_phase_start()
        self.prior_target = None
        self.prior_matrix = None
        self.warper = self.trainer.warper
        
    def _update_target(self, id, feat, matrix):
        self.prior_target = feat
        self.prior_matrix = matrix
    
    
    def on_iteration(self, iteration: int):
        
        last_cam, curr_cam = self._get_viewpoint_cam()
        
        with self.trainer.timer as timer:
            
            
            self.render_pkg = self.trainer.get_render_pkgs(curr_cam)
            render_image = self.render_pkg["render"]
            curr_image = self.trainer.ctx.scene_images[curr_cam.uid]
            
            self.render_feat = self.feature_extractor(render_image)
            curr_scene_features_mask = self.trainer.ctx.scene_features_mask_list[curr_cam.uid]
            curr_fusion_mask = self.trainer.ctx.fusion_masks[curr_cam.uid]
            depth_group_num = self.trainer.config.style.depth_group_num
            render_features_list = get_separated_list(self.render_feat, 
                                                      curr_scene_features_mask,
                                                      depth_group_num)
            
            # Calculate depth loss
            render_depth = self.render_pkg["depth"]
            curr_depth = self.trainer.ctx.depth_images[curr_cam.uid]
            depth_loss = torch.mean((render_depth - curr_depth) ** 2)
            fc, fh, fw = self.render_feat.shape
            diff_matrix = torch.zeros((1, fh, fw), device=self.render_feat.device)

            if last_cam is None:
                # TODO：一个更好的先验控制模块
                # input: 给定reference图的特征图 or mask+对应区域特征图角度控制
                # TODO: mask额外接入or单独接入
                # TODO: 跑NNST获得reference图
                
                theta = self.trainer.config.style.theta
                
                fc, fh, fw = self.render_feat.shape
                target_feat = torch.zeros((fc, fh, fw), device=self.render_feat.device)
                target_matrix = torch.zeros((1, fh, fw), device=self.render_feat.device)
                
                for i in range(depth_group_num):
                    mask = (curr_scene_features_mask == i)
                    if mask.sum() == 0:
                        continue
                    
                    style_feat = self.trainer.ctx.style_features_list[i]
                    style_matrix = self.trainer.ctx.style_matrix_list[i]
                    scene_features = self.trainer.ctx.scene_features_list[curr_cam.uid][i]
                    
                    num_clusters = style_feat.shape[1] // 360
                    A = style_feat[:, theta * num_clusters: (theta + 1) * num_clusters]
                    A_mat = style_matrix[:, theta * num_clusters: (theta + 1) * num_clusters]
                    
                    A_indices = torch.randint(0, num_clusters - 1, 
                                              (fc, scene_features.shape[1]), device=style_feat.device)
                    A_mat_indices = torch.randint(0, num_clusters - 1, 
                                                  (1, scene_features.shape[1]), device=style_feat.device)
                    
                    target_feat[:, mask] = torch.gather(A, 1, A_indices)
                    target_matrix[:, mask] = torch.gather(A_mat, 1, A_mat_indices)
                    
                    
                consistent_loss = 0
            else:
                
                
                last_image = self.trainer.ctx.scene_images[last_cam.uid]

                with torch.no_grad():
                    K = torch.from_numpy(last_cam.K.astype(np.float32)).unsqueeze(0)
                    
                    transformation1 = torch.from_numpy(curr_cam.transformation).unsqueeze(0)
                    transformation2 = torch.from_numpy(last_cam.transformation).unsqueeze(0)
                    
                    warped_frame2, mask2, pos_pre = self.warper.forward_warp(
                        curr_image.unsqueeze(0), None,
                        curr_depth.unsqueeze(0).unsqueeze(0), 
                        transformation1, transformation2, K, None
                    )
                    # mask2: [1, 1, h, w]
                    warped_frame2 = warped_frame2.squeeze(0)
                    
                    diff = torch.sqrt(torch.sum((warped_frame2 - last_image) ** 2, dim=0))  # 沿着通道维度计算欧几里得距离
                    
                    threshold = 0.3
                    # 生成 mask：如果颜色差异小于阈值，mask 为 1，否则为 0
                    trans_mask = torch.logical_and(diff < threshold, mask2.squeeze()).int()

                    # TODO: 动态维护(done)
                    prior_target = self.prior_target
                    # TODO：按照几何进行改变(done-旋转)
                    # 使用4个点对（or更多）的关系求解旋转和错切
                    prior_matrix = self.prior_matrix
                    
                    h, w = trans_mask.shape
                    rows = torch.arange(h, device=prior_target.device)  # (h,)
                    cols = torch.arange(w, device=prior_target.device)  # (w,)
                    grid_i, grid_j = torch.meshgrid(rows, cols, indexing="ij")
                    # 原始像素坐标矩阵
                    # pos_pre为变换到前一个像素的
                    # 记得转成float
                    pos = torch.stack([grid_i, grid_j], dim=-1)
                    
                    # 处理先验mask和对应问题
                    _, fh, fw = self.render_feat.shape
                    # 特征图坐标
                    fh_grid, fw_grid = torch.meshgrid(
                        torch.arange(fh, device=prior_target.device),
                        torch.arange(fw, device=prior_target.device),
                        indexing='ij'
                    )
                    # 特征像素对应的原始像素的中心
                    img_h = fh_grid * 8 + 4
                    img_w = fw_grid * 8 + 4
                    # TODO：转换成特征像素对应的多个原始像素（最少四个？）
                    # TODO：计算warp前对应的原始像素(done-pos_float)
                    # TODO：计算其仿射变换矩阵（由于错切难以处理，可以考虑简化为只有旋转）(done)
                    # TODO: 用仿射变换矩阵修改先验矩阵(done-原始角度+新角度)
                    
                    # TODO: 解开注释
                    rotation = compute_rotation_angles(pos_pre, pos, fh, fw)
                    prior_matrix = (prior_matrix + rotation) % 360
                    
                    # 获得先验mask, 构建特征图大小的mask
                    prior_idx = torch.zeros(fh, fw, 2).to(prior_target.device)
                    prior_idx = pos[img_h, img_w]
                    rows = prior_idx[:, :, 0]
                    cols = prior_idx[:, :, 1]
                    prior_mask = trans_mask[rows, cols]
                    
                    
                    prior_idx = (pos_pre[img_h, img_w] - 4.0) / 8.0
                    # 转换成
                    grid = prior_idx.unsqueeze(0)
                    
                    prior_feats = F.grid_sample(prior_target.unsqueeze(0), grid, mode='bilinear', align_corners=False)
                    prior_feats = prior_feats.squeeze()
                # mask_feats = mask[]
                
                # 计算新的target_feats
                # 其实就是找一个index, 这个index的特征和原图，先验特征和先验矩阵的总相似情况的argmin
                # 需要注意的是mask，即有先验才用先验，没有先验则和原来nnfm一样（除了if else如何有好的实现？， 应该是要修改求argmin的时候先验项的系数，mask为0时置0）
                    
                    # TODO: 动态维护；可切换分支（使用先验or不使用先验）
                    
                    prior_mask_list = get_separated_list(prior_mask.unsqueeze(0), 
                                                         curr_scene_features_mask,
                                                         depth_group_num)
                    prior_feature_list = get_separated_list(prior_feats, 
                                                            curr_scene_features_mask,
                                                            depth_group_num)
                    prior_matrix_list = get_separated_list(prior_matrix, 
                                                           curr_scene_features_mask,
                                                           depth_group_num)
                    
                    fc, fh, fw = self.render_feat.shape
                    target_feat = torch.zeros((fc, fh, fw), device=self.render_feat.device)
                    target_matrix = torch.zeros((1, fh, fw), device=self.render_feat.device)
                    
                    # ic(prior_mask.shape, prior_feats.shape, prior_matrix.shape)
                    # exit(0)
                    for i in range(depth_group_num):
                        mask = (curr_scene_features_mask == i)
                        if mask.sum() == 0:
                            continue
                        
                        style_feat = self.trainer.ctx.style_features_list[i]
                        style_matrix = self.trainer.ctx.style_matrix_list[i]
                        p_mask = prior_mask_list[i]
                        p_feat = prior_feature_list[i]
                        p_matrix = prior_matrix_list[i]
                        
                        render_feat = render_features_list[i]
                        
                        # ic(style_feat.shape, style_matrix.shape, render_feat.shape, (curr_scene_features_mask == i).sum())

                        # TODO: calculate nnfm target_matrix to get distort level
                        curr_target_feat, curr_target_matrix, diff = prior_feat_replace(render_feat, 
                                                                                  style_feat, 
                                                                                  style_matrix,
                                                                                  p_mask,
                                                                                  p_feat, 
                                                                                  p_matrix,
                                                                                  flag=self.config.style.enable_weighted)
                        target_feat[:, mask] = curr_target_feat
                        target_matrix[:, mask] = curr_target_matrix
                        if self.config.style.enable_weighted:
                            diff_matrix[:, mask] = diff
                        
                
            self._update_target(curr_cam.uid, target_feat, target_matrix)
            
            # TODO: weighted
            
            weighted_matrix = (90.0 - diff_matrix) / 90.0
            if self.config.style.enable_weighted:
                weighted_matrix = self.trainer.config.style.lambda_adaptive * (curr_fusion_mask.unsqueeze(0) + weighted_matrix * 0.25)
            # weighted_matrix = weighted_matrix * curr_fusion_mask.unsqueeze(0)
            prior_loss = cos_distance(target_feat, self.render_feat, weighted_matrix)
            content_loss = content_loss_fn(render_features_list, 
                                           self.trainer.ctx.scene_features_list[curr_cam.uid], weighted_matrix)
            
            # content_loss = torch.mean((self.render_feat - self.original_feats[curr_cam.uid]) ** 2) 
            
            top2_values, _ = torch.topk(self.trainer.gaussians.get_scaling, k=2, dim=1) 
            shape_loss = (top2_values[:, 0] / top2_values[:, 1]).mean()
            imgtv_loss = get_imgtv_loss(render_image)
            
            # concat_and_save_images("./image.jpg", render_image, render_depth)
            # if not self.trainer.config.style.densify:
            # ic(loss_delta_opacity, loss_delta_scaling)
            # exit(0)
            # ic(loss_delta_scaling, loss_delta_opacity)
            
            consistent_loss = 0
            loss = (
                # Todo: check consistent_loss
                self.trainer.config.style.lambda_consistent_loss * consistent_loss
                + self.trainer.config.style.lambda_prior_loss * prior_loss
                + self.trainer.config.style.lambda_content_loss * content_loss
                + self.trainer.config.style.lambda_imgtv_loss * imgtv_loss
                + self.trainer.config.style.lambda_depth_loss * depth_loss
                + self.trainer.config.style.lambda_shape_loss * shape_loss
            )
            
            # loss_delta_opacity = torch.norm(self.trainer.gaussians._opacity - self.trainer.gaussians.parent_opacity)
            # loss_delta_scaling = torch.norm(self.trainer.gaussians._scaling - self.trainer.gaussians.parent_scaling)
            # loss_delta_position = torch.norm(self.trainer.gaussians._xyz - self.trainer.gaussians.parent_position)
            # if not self.trainer.config.style.densify:
            # loss += (self.trainer.config.style.lambda_delta_opacity * loss_delta_opacity 
            #         + self.trainer.config.style.lambda_delta_scaling * loss_delta_scaling)
            
            self.update(iteration, loss)
            
            if self.trainer.config.app.need_log:
                wandb.log({
                    "Loss": loss.item(),
                    "Prior Loss": prior_loss.item(),
                    "Content Loss": content_loss.item(),
                    "ImgTV Loss": imgtv_loss.item(),
                    "Depth Loss": depth_loss.item(),
                    "Shape Loss": shape_loss.item()
                })
            
        
        return {
            "Points": f"{self.trainer.gaussians._opacity.shape[0]}",
            "Loss": f"{loss.item():.{7}f}"
        }, timer.elapsed_ms

    def _get_viewpoint_cam(self):
        if not self.viewpoint_stack:
            self.viewpoint_stack = self.trainer.scene.getTrainCameras()
            self.viewpoint_len = len(self.viewpoint_stack)
            self.viewpoint_idx = -1
        self.viewpoint_idx = (self.viewpoint_idx + 1) % self.viewpoint_len
        
        last_cam = None
        if self.viewpoint_idx != 0:
            last_cam = self.viewpoint_stack[self.viewpoint_idx - 1]
        curr_cam = self.viewpoint_stack[self.viewpoint_idx]
        return last_cam, curr_cam