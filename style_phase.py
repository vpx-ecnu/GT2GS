# style_phase.py
from abc import ABC, abstractmethod
from enum import Enum, auto
from typing import Dict
import time 
from random import randint
import torch
from style_utils import *
from utils.loss_utils import l1_loss, ssim
from style_loss import *
from torch.linalg import svd, det 
import wandb
    

class TrainingPhase(ABC):
    def __init__(self, trainer, uid, name, start_iter, end_iter):
        self.trainer = trainer
        self.config = trainer.config
        self.start_iter = start_iter
        self.end_iter = end_iter
        self.viewpoint_stack = None
        self.name = name
        self.uid = uid
        
    def update(self, iteration, loss):
        
        loss.backward()
        self._densification(iteration)
        self.trainer.gaussians.optimizer.step()
        self.trainer.gaussians.optimizer.zero_grad(set_to_none=True)

    def on_phase_start(self): ...

    @abstractmethod
    def on_iteration(self, iteration: int) -> Dict[str, torch.Tensor]: ...

    def on_phase_end(self): ...
    
    def _densification(self, iteration: int): ...


class ProcessPhase(TrainingPhase):
    
    def on_iteration(self, iteration: int) -> Dict[str, torch.Tensor]:
        
        viewpoint_cam = self._get_viewpoint_cam()
        # viewpoint_cam = self.trainer.scene.getTrainCameras()[0]
        
        with self.trainer.timer as timer:
            
            self.render_pkg = self.trainer.get_render_pkgs(viewpoint_cam)
            render_image = self.render_pkg["render"]
            render_depth = self.render_pkg["depth"]
            original_image = self.trainer.ctx.original_images[viewpoint_cam.uid]
            curr_depth = self.trainer.ctx.depth_images[viewpoint_cam.uid]
            
            # render_depth = self.render_pkg["depth"]
            # original_depth = self.trainer.ctx.depth_images[viewpoint_cam.uid]
            # concat_and_save_images("./image.jpg", original_image, render_image, original_depth, render_depth)

            
            Ll1 = l1_loss(render_image, original_image)
            ssim_val = ssim(render_image, original_image)
            
            concat_and_save_images("./image.jpg", original_image, render_image, curr_depth, render_depth)  
            loss = (
                (1.0 - self.trainer.config.opt.lambda_dssim) * Ll1 + 
                self.trainer.config.opt.lambda_dssim * (1.0 - ssim_val)
            )
            self.update(iteration, loss)
              
        return {
            "Points": f"{self.trainer.gaussians._opacity.shape[0]}",
            "Loss": f"{loss.item():.{7}f}"
        }, timer.elapsed_ms
    
    def _get_viewpoint_cam(self):
        if not self.viewpoint_stack or len(self.viewpoint_stack) == 0:
            self.viewpoint_stack = self.trainer.scene.getTrainCameras().copy()
        return self.viewpoint_stack.pop(randint(0, len(self.viewpoint_stack) - 1))

            
class PreProcessPhase(ProcessPhase):
    
    
    def _densification(self, iteration: int):
        
        gaussians = self.trainer.gaussians
        opt = self.trainer.config.opt
        # scene = self.trainer.scene
        # dataset = self.trainer.config.model
        
        # viewspace_point_tensor = self.render_pkg["viewspace_points"]
        # visibility_filter = self.render_pkg["visibility_filter"]
        # radii = self.render_pkg["radii"]
        
        # gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
        # gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

        # if iteration % opt.densification_interval == 0:
        #     size_threshold = 20 if iteration > opt.opacity_reset_interval else None
        #     gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)
            
        # if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == self.start_iter):
        #     gaussians.reset_opacity()
        
        if (
            (iteration - self.start_iter) % opt.densification_interval == 0 and 
            (iteration - self.start_iter) <= (self.end_iter - self.start_iter + 1) // 2
        ):
            tmp = torch.max(gaussians.get_scaling, dim=1).values
            self.threshold = torch.quantile(tmp, 0.95)
            gaussians.split_special_gaussians(tmp > self.threshold)
            
    @torch.no_grad
    def on_phase_start(self):
        if self.config.style.color_transfer:
            color_transfer(self.trainer.ctx)
            
    @torch.no_grad
    def on_phase_end(self):
        self.trainer.warper = Warper()
        self.trainer.feat_extractor = FeatureExtractor()
        self._init_original_feats()
        self._init_style_feat()
        
        
        viewpoint_stack = self.trainer.scene.getTrainCameras()
        self.trainer.ctx.depth_images = []
        # self.trainer.ctx.original_images = []
        
        # colmap maybe change image's size
        min_h, min_w = 10000, 10000
        for i, view in enumerate(viewpoint_stack):
            min_h = min(min_h, view.image_height)
            min_w = min(min_w, view.image_width)
        self.trainer.ctx.image_width = min_w
        self.trainer.ctx.image_height = min_h
        
        for _, view in enumerate(viewpoint_stack):
            depth_image = self.trainer.get_render_pkgs(view)["depth"]
            self.trainer.ctx.depth_images.append(depth_image.squeeze().detach())
            
        self.trainer.ctx.depth_images = torch.stack(self.trainer.ctx.depth_images).to(device=self.trainer.device)
 
    def _init_original_feats(self):
        self.trainer.ctx.original_feats = []
        
        for i, original_image in enumerate(self.trainer.ctx.original_images):
            self.trainer.ctx.original_feats.append(self.trainer.feat_extractor.get_features(original_image))
    
        self.trainer.ctx.original_feats = torch.stack(self.trainer.ctx.original_feats)
    
    def _init_style_feat(self):
        
        self.trainer.ctx.style_feat = []
        self.trainer.ctx.style_matrix = []
        
        style_image = self.trainer.ctx.style_image
        _, style_img_width, style_img_height = style_image.shape
        
        # TODO: 
        # 1. 考虑用clip替换聚类的pipeline
        # 2. 对齐ref-npr的setting，可以有一个完全先验的特征图
        # 3. ref-npr做实验，验证没有rgb loss，只有特征图loss情况下的结果
        for i in range(0, 360):
            # 先假设错切参数为0，旋转角度由i得到
            Hx = Hy = 0
            # TODO: 要保证是float格式
            theta = i * 1.0
            
            # 获得线性变换矩阵（包括旋转角度和错切参数）
            M, M_parameter = generate_transformation_matrix(theta, Hx, Hy, style_img_width, style_img_height)
            
            # 根据i获得增强后的图片
            # new_image = F.affine(style_image, theta, [0,9], 1.0, [Hx, Hy], resample=Image.BICUBIC)
            new_image = tensor_img_transformation(style_image, M, i)
            
            # extract vgg feature
            # self.style_feats.append(self.get_feats(style_image.unsqueeze(0)))
            img_feats = self.trainer.feat_extractor.get_features(new_image, False)
            # print(img_feats.shape)
            ######################################################################
            # 不聚类
            if self.trainer.config.style.gta_type == "default":
                c, h, w = img_feats.shape
                img_feats = img_feats.view(c, -1)
                _, num_clusters = img_feats.shape
            
            ######################################################################
            # 聚类
            if self.trainer.config.style.gta_type == "kmeans":
                # 处理为k-means可用形式
                c, h, w = img_feats.shape
                img_feats_flat = img_feats.permute(1, 2, 0).reshape(-1, c)
                img_feats_np = img_feats_flat.cpu().numpy()
                
                # k-means
                # TODO: 超参调整
                num_clusters = 40
                kmeans = KMeans(n_clusters=num_clusters, random_state=42, max_iter=300)
                kmeans.fit(img_feats_np)
                
                # 获取每个像素对应的聚类标签
                cluster_labels = kmeans.labels_  # [h * w]

                # 获取聚类中心
                cluster_centers = kmeans.cluster_centers_  # [num_clusters, c]

                # 将每个像素替换为其所属聚类的中心向量
                discretized_feats_flat = torch.tensor(cluster_centers, device=img_feats.device)

                # 将特征还原回原始形状 [c, -1]
                img_feats = discretized_feats_flat.view(-1, c).permute(1, 0)
            
            ######################################################################
            # 裁剪
            # TODO: 后面可以手动控制变量范围
            if self.trainer.config.style.gta_type == "clip":
                c, h, w = img_feats.shape
                target_h = 8
                target_w = 8
                
                start_h = (h - target_h) // 2
                start_w = (w - target_w) // 2
                
                cropped_tensor = img_feats[:, start_h:start_h + target_h, start_w:start_w + target_w]
                # view必须张量在内存中连续，所以用reshape
                img_feats = cropped_tensor.reshape(c, -1)
                _, num_clusters = img_feats.shape
            
            # 将新特征组加入总特征集合（当前未考虑深度分组）
            
            self.trainer.ctx.style_feat.append(img_feats)
            
            # M_tensor = torch.from_numpy(M_parameter)
            # 目前先只存了旋转角度，用于求loss，而不是
            M_tensor = torch.tensor(theta)
            matrix_list = torch.stack([M_tensor] * num_clusters).to("cuda").unsqueeze(0)
            self.trainer.ctx.style_matrix.append(matrix_list)

        
        self.trainer.ctx.style_feat = torch.cat(self.trainer.ctx.style_feat, dim=1)
        self.trainer.ctx.style_matrix = torch.cat(self.trainer.ctx.style_matrix, dim=1)
        

class PostProcessPhase(ProcessPhase):
    
    @torch.no_grad
    def on_phase_start(self):
        
        self.viewpoint_stack = self.trainer.scene.getTrainCameras().copy()
        self.viewpoint_len = len(self.viewpoint_stack)
        for i in range(0, self.viewpoint_len):
            curr_cam = self.viewpoint_stack[i]
            self.render_pkg = self.trainer.get_render_pkgs(curr_cam)
            render_img = self.render_pkg["render"]
            self.trainer.ctx.original_images[curr_cam.uid] = render_img
            
        if self.config.style.color_transfer:
            color_transfer(self.trainer.ctx)
        
        

class StylizationPhase(TrainingPhase):
    
    def _densification(self, iteration: int):
        if not self.trainer.config.style.density:
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
            
        if (iteration - self.start_iter) % opt.style_densification_interval == 0:
            # threshold = torch.quantile(radii[visibility_filter].float(), 0.10)
            
            # big_mask = torch.logical_and(visibility_filter, radii > threshold)
            # gaussians.split_special_gaussians(big_mask, 10)
            
            # tmp = torch.max(gaussians.get_scaling, dim=1).values
            # threshold = torch.quantile(tmp, 0.80)
            # gaussians.split_special_gaussians(tmp > threshold)
            gaussians.densify_and_prune(opt.style_densification_threshold, 0.005, scene.cameras_extent, 20)
    
    @torch.no_grad
    def on_phase_start(self):
        # self.target_feats = {}
        # self.target_matrixs = {}
        # self.projection = {}
        self.prior_target = None
        self.prior_matrix = None
        self.warper = self.trainer.warper
        self.feat_extractor = self.trainer.feat_extractor
        self.original_feats = self.trainer.ctx.original_feats
        self.style_feat = self.trainer.ctx.style_feat
        self.style_matrix = self.trainer.ctx.style_matrix
        if self.trainer.config.style.no_grad:
            self.trainer.gaussians._xyz.requires_grad_(False)
            self.trainer.gaussians._scaling.requires_grad_(False)
            self.trainer.gaussians._opacity.requires_grad_(False)
        # self._init_projection()
    
    def on_phase_end(self):
        # pass
        if self.trainer.config.style.no_grad:
            self.trainer.gaussians._xyz.requires_grad_(True)
            self.trainer.gaussians._scaling.requires_grad_(True)
            self.trainer.gaussians._opacity.requires_grad_(True)
        else:
            pass
    
    def _update_target(self, id, feat, matrix):
        self.prior_target = feat
        self.prior_matrix = matrix
    
    def on_iteration(self, iteration: int) -> Dict[str, torch.Tensor]:
 
        def compute_rotation_angles(A, B, fh, fw):
            """
            计算特征图中每个位置的旋转角度（并行化版本）。
            
            参数：
                A (torch.Tensor): 形状为 [h, w, 2] 的张量，表示原始坐标 [X, Y]
                B (torch.Tensor): 形状为 [h, w, 2] 的张量，表示变换后的坐标 [X, Y]
                fh (int): 特征图的高度
                fw (int): 特征图的宽度
            
            返回：
                C (torch.Tensor): 形状为 [1, fh, fw] 的张量，表示每个特征图像素的旋转角度
            """
            # 获取原图像尺寸
            h, w, _ = A.shape
            
            # 计算池化步幅
            pool_size_h = h // fh
            pool_size_w = w // fw
            
            # 确保尺寸匹配
            # assert h % fh == 0 and w % fw == 0, "特征图尺寸必须能整除原图像尺寸"
            
            # 生成所有 (x, y) 位置的网格
            x_indices = torch.arange(fh, device=A.device)
            y_indices = torch.arange(fw, device=A.device)
            x_grid, y_grid = torch.meshgrid(x_indices, y_indices, indexing='ij')
            
            # 计算四个角点的索引
            top_left_x = x_grid * pool_size_h
            top_left_y = y_grid * pool_size_w
            top_right_x = x_grid * pool_size_h
            top_right_y = (y_grid + 1) * pool_size_w - 1
            bottom_left_x = (x_grid + 1) * pool_size_h - 1
            bottom_left_y = y_grid * pool_size_w
            bottom_right_x = (x_grid + 1) * pool_size_h - 1
            bottom_right_y = (y_grid + 1) * pool_size_w - 1
            
            # 提取所有位置的角点，形状为 (fh, fw, 4, 2)
            A_S = torch.stack([
                A[top_left_x, top_left_y],
                A[top_right_x, top_right_y],
                A[bottom_left_x, bottom_left_y],
                A[bottom_right_x, bottom_right_y]
            ], dim=2)
            
            B_S = torch.stack([
                B[top_left_x, top_left_y],
                B[top_right_x, top_right_y],
                B[bottom_left_x, bottom_left_y],
                B[bottom_right_x, bottom_right_y]
            ], dim=2)
            
            # 批量中心化
            mu_a = A_S.mean(dim=2, keepdim=True)  # 形状 (fh, fw, 1, 2)
            mu_b = B_S.mean(dim=2, keepdim=True)  # 形状 (fh, fw, 1, 2)
            A_centered = A_S - mu_a  # 形状 (fh, fw, 4, 2)
            B_centered = B_S - mu_b  # 形状 (fh, fw, 4, 2)
            
            # 批量构造 H 矩阵
            H = torch.einsum('fhij,fhjk->fhik', B_centered.permute(0, 1, 3, 2), A_centered)
            
            # 批量 SVD 分解
            U, S, Vh = torch.svd(H)  # U, Vh: (fh, fw, 2, 2), S: (fh, fw, 2)
            
            # 计算旋转矩阵 R = U @ Vh^T
            R = torch.einsum('fhij,fhjk->fhik', U, Vh.permute(0, 1, 3, 2))  # (fh, fw, 2, 2)
            
            # 检查行列式并调整（确保 R 是旋转矩阵）
            det_R = R[:, :, 0, 0] * R[:, :, 1, 1] - R[:, :, 0, 1] * R[:, :, 1, 0]  # (fh, fw)
            mask = det_R < 0
            if mask.any():
                U[mask, :, -1] = -U[mask, :, -1]  # 调整 U 的最后一列
                R[mask] = torch.bmm(U[mask], Vh[mask].permute(0, 2, 1))
            
            # 批量提取角度 theta
            theta = torch.atan2(R[:, :, 1, 0], R[:, :, 0, 0])  # 形状 (fh, fw)
            
            # 赋值给 C
            C = theta.unsqueeze(0)  # 形状 (1, fh, fw)
            
            return C
        
        # def compute_rotation_angles(A, B, fh, fw):
        #     h, w, _ = A.shape
            
        #     # pool_size = 8
        #     pool_size = h // fh
            
        #     C = torch.zeros(1, fh, fw, device=A.device, dtype=A.dtype)
            
        #     for x in range(fh):
        #             for y in range(fw):
        #                 # 提取对应区域点对
        #                 # A_S = A[x * pool_size : (x + 1) * pool_size,
        #                 #         y * pool_size : (y + 1) * pool_size, :]
        #                 # B_S = B[x * pool_size : (x + 1) * pool_size,
        #                 #         y * pool_size : (y + 1) * pool_size, :]
                        
        #                 # 只提取角点进行加速
        #                 top_left = (x * pool_size, y * pool_size)
        #                 top_right = (x * pool_size, (y + 1) * pool_size - 1)
        #                 bottom_left = ((x + 1) * pool_size - 1, y * pool_size)
        #                 bottom_right = ((x + 1) * pool_size - 1, (y + 1) * pool_size - 1)
                        
        #                 # 从 A 和 B 中提取四个角点
        #                 A_S = torch.stack([
        #                     A[top_left[0], top_left[1]],
        #                     A[top_right[0], top_right[1]],
        #                     A[bottom_left[0], bottom_left[1]],
        #                     A[bottom_right[0], bottom_right[1]]
        #                 ])  # 形状 (4, 2)
                        
        #                 B_S = torch.stack([
        #                     B[top_left[0], top_left[1]],
        #                     B[top_right[0], top_right[1]],
        #                     B[bottom_left[0], bottom_left[1]],
        #                     B[bottom_right[0], bottom_right[1]]
        #                 ])
                        
                        
                        
        #                 # 不考虑平移，考虑以中心进行的2*2变换
        #                 mu_a = A_S.mean(dim=[0, 1])  # 形状 (2,)
        #                 mu_b = B_S.mean(dim=[0, 1])
                        
        #                 A_centered = A_S - mu_a
        #                 B_centered = B_S - mu_b
                        
        #                 # 估计矩阵
        #                 H = (B_centered.reshape(-1, 2).t() @ A_centered.reshape(-1, 2))
                        
        #                 # SVD（分解旋转和缩放）
        #                 U, S, Vh = svd(H)
                        
        #                 # 旋转矩阵
        #                 R = U @ Vh.t()
                        
        #                 # 检查行列式，确保 R 是旋转矩阵（det=1）而非反射（det=-1）
        #                 if det(R) < 0:
        #                     U[:, -1] = -U[:, -1]
        #                     R = U @ Vh.t()
                        
        #                 # 从旋转矩阵提取角度 theta
        #                 theta = torch.atan2(R[1, 0], R[0, 0])
                        
        #                 C[0, x, y] = theta
    
        #     return C
        
        last_cam, curr_cam = self._get_viewpoint_cam()
        
        with self.trainer.timer as timer:
            
            self.render_pkg = self.trainer.get_render_pkgs(curr_cam)
            render_image = self.render_pkg["render"]
            curr_image = self.trainer.ctx.original_images[curr_cam.uid]
            self.render_feat = self.feat_extractor.get_features(render_image)
            
            
            render_depth = self.render_pkg["depth"]
            curr_depth = self.trainer.ctx.depth_images[curr_cam.uid]
            depth_loss = torch.mean((render_depth - curr_depth) ** 2)
            
            if self.trainer.config.style.prior == False or self.trainer.cur_phase % 2 == 0:

                target_feat, target_matrix = nnfm_feat_replace(self.render_feat, self.style_feat, self.style_matrix)
                consistent_loss = 0
            elif last_cam is None:
                # TODO：一个更好的先验控制模块
                # input: 给定reference图的特征图 or mask+对应区域特征图角度控制
                # TODO: mask额外接入or单独接入
                # TODO: 跑NNST获得reference图
                # if self.trainer.config.style.ref_image:
                #     target_feat = 
                #     target_matrix = 
                # elif self.trainer.config.style.ref_mask:
                #     target_feat = 
                #     target_matrix = 
                # else:
                #     target_feat = 
                #     target_matrix = 
                    # pass


                # self.feat_extractor.get_features(ref_image)
                tmp_val = 64
                A, A_mat = self.style_feat[:, :tmp_val], self.style_matrix[:, :tmp_val]
                B, B_mat = self.style_feat[:, tmp_val*90:tmp_val*91], self.style_matrix[:, tmp_val*90:tmp_val*91]
                
                _, fc, fh, fw = self.original_feats.shape
                C = torch.zeros(fc, fh, fw).to("cuda")
                matc, _ = A_mat.shape
                C_mat = torch.zeros(matc, fh, fw).to("cuda")
                A_indices = torch.randint(0, 5, (fc, fh // 2, fw))
                for i in range(fc):
                    C[i, :fh // 2, :] = A[i, A_indices[i]]
                for i in range(matc):
                    C_mat[i, :fh // 2, :] = A_mat[i, A_indices[i]]
                
                B_indices = torch.randint(0, 5, (fc, fh - fh // 2, fw))
                for i in range(fc):
                    C[i, fh // 2:, :] = B[i, B_indices[i]]
                for i in range(matc):
                    C_mat[i, fh // 2:, :] = B_mat[i, B_indices[i]]
                # A_expanded = setA.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, fh, fw)
                # B_expanded = setB.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, fh, fw)
                # upper_half = A_expanded[:, :h//2, :, :]
                # lower_half = B_expanded[:, h//2:, :, :]
                # C = torch.cat((upper_half, lower_half), dim=1)
                target_feat = C
                target_matrix = C_mat
                consistent_loss = 0
                
            else:
                
                
                last_image = self.trainer.ctx.original_images[last_cam.uid]

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
                    trans_mask = (diff < threshold).float()
                trans_image = trans_mask * warped_frame2 + (1 - trans_mask) * curr_image
                    
                trans_feat = self.feat_extractor.get_features(trans_image)
                
                # TODO: 动态维护
                consistent_loss = cos_distance(self.prior_target, trans_feat)
                    ############################
                    
                    # 读入先验数据
                    # prior_target和prior_matrix还需要进行双线性插值进行转换才可以使用
                with torch.no_grad():
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
                    pos_float = pos * 1.0
                    
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
                    
                    # rotation = compute_rotation_angles(pos_pre, pos_float, fh, fw)
                    # prior_matrix = prior_matrix + rotation
                    
                    
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
                    target_feat, target_matrix = prior_feat_replace(
                        self.render_feat, self.style_feat, self.style_matrix,
                        prior_mask, prior_feats, prior_matrix
                    )
            
                # print("111:", target_feat.mean(), target_matrix.mean())
                # exit(0)
            
            
                # trans_mask = self.trans_mask[(last_cam.uid, curr_cam.uid)]
                # trans_depth = self.trans_depth[(last_cam.uid, curr_cam.uid)]
                # trans_pos = self.trans_pos[(last_cam.uid, curr_cam.uid)]
            
                
            self._update_target(curr_cam.uid, target_feat, target_matrix)
            prior_loss = cos_distance(target_feat, self.render_feat)
            content_loss = torch.mean((self.render_feat - self.original_feats[curr_cam.uid]) ** 2) 
            
            top2_values, _ = torch.topk(self.trainer.gaussians.get_scaling, k=2, dim=1) 
            shape_loss = (top2_values[:, 0] / top2_values[:, 1]).mean()
            imgtv_loss = get_imgtv_loss(render_image)
            
            concat_and_save_images("./image.jpg", curr_image, render_image, curr_depth, render_depth)

            
            
            loss = (
                self.trainer.config.style.lambda_consistent_loss * consistent_loss
                + self.trainer.config.style.lambda_prior_loss * prior_loss
                + self.trainer.config.style.lambda_content_loss * content_loss
                + self.trainer.config.style.lambda_imgtv_loss * imgtv_loss
                + self.trainer.config.style.lambda_depth_loss * depth_loss
                + self.trainer.config.style.lambda_shape_loss * shape_loss
            )
            
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

           
class GeometryProtectPhase(TrainingPhase):
    
    @torch.no_grad
    def on_phase_start(self):
        self.trainer.gaussians._features_dc.requires_grad_(False)
        self.trainer.gaussians._features_rest.requires_grad_(False)
        self.feat_extractor = self.trainer.feat_extractor
        # if self.config.style.color_transfer:
        #     color_transfer(self.trainer.ctx)

        # render_ctx(self.trainer.ctx)
        # exit(0)
    
    def on_phase_end(self):
        self.trainer.gaussians._features_dc.requires_grad_(True)
        self.trainer.gaussians._features_rest.requires_grad_(True)

    def on_iteration(self, iteration: int) -> Dict[str, torch.Tensor]:
        
        viewpoint_cam = self._get_viewpoint_cam()
        # viewpoint_cam = self.trainer.scene.getTrainCameras()[0]
        
        # TODO: 加入对参数梯度的控制，锁颜色
        
        with self.trainer.timer as timer:
            
            self.render_pkg = self.trainer.get_render_pkgs(viewpoint_cam)
            render_image = self.render_pkg["render"]
            curr_image = self.trainer.ctx.original_images[viewpoint_cam.uid]
            render_depth = self.render_pkg["depth"]
            curr_depth = self.trainer.ctx.depth_images[viewpoint_cam.uid]
            
            depth_loss = torch.mean((render_depth - curr_depth) ** 2)
            render_feat = self.feat_extractor.get_features(render_image)
            content_feat = self.feat_extractor.get_features(curr_image)
            content_loss = torch.mean((render_feat - content_feat) ** 2) 
            
            # render_depth = self.render_pkg["depth"]
            # original_depth = self.trainer.ctx.depth_images[viewpoint_cam.uid]
            # concat_and_save_images("./image.jpg", original_image, render_image, original_depth, render_depth)

            # TODO：1.depth loss 2.content loss?
            
            loss = (
                self.trainer.config.style.lambda_depth_loss * depth_loss 
                + self.trainer.config.style.lambda_content_loss * content_loss
            )
            
            self.update(iteration, loss)
            
        return {
            "Points": f"{self.trainer.gaussians._opacity.shape[0]}",
            "Loss": f"{loss.item():.{7}f}"
        }, timer.elapsed_ms
    
    def _get_viewpoint_cam(self):
        if not self.viewpoint_stack:
            self.viewpoint_stack = self.trainer.scene.getTrainCameras().copy()
        return self.viewpoint_stack.pop(randint(0, len(self.viewpoint_stack) - 1))

    def _densification(self, iteration: int):
        return 
        # gaussians = self.trainer.gaussians
        # opt = self.trainer.config.opt
        # scene = self.trainer.scene
        # dataset = self.trainer.config.model
        
        # viewspace_point_tensor = self.render_pkg["viewspace_points"]
        # visibility_filter = self.render_pkg["visibility_filter"]
        # radii = self.render_pkg["radii"]
        
        # gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
        # gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

        # if iteration % opt.densification_interval == 0:
        #     size_threshold = 20 if iteration > opt.opacity_reset_interval else None
        #     gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)
        
        # if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == self.start_iter):
        #     gaussians.reset_opacity()           