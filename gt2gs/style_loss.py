# style_loss.py
import torch
import torchvision
from torchvision.models import VGG16_Weights
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
from dataclasses import dataclass
from gt2gs.style_utils import *

# @dataclass
# class LossContext:
    

def get_imgtv_loss(image):
    image = image.unsqueeze(0)
    w_variance = torch.mean(torch.pow(image[:, :, :-1] - image[:, :, 1:], 2))
    h_variance = torch.mean(torch.pow(image[:, :-1, :] - image[:, 1:, :], 2))
    img_tv_loss = (h_variance + w_variance) / 2.0
    return img_tv_loss


class FeatureExtractor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
        self.vgg = torchvision.models.vgg16(weights=VGG16_Weights.DEFAULT).eval().to("cuda")
        self.feature_layers = [11, 13, 15]
        
        self.normalize = torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    
    def __call__(self, original: torch.Tensor, downscale=True) -> torch.Tensor:
        image = original.unsqueeze(0)
        
        if downscale:
            image = F.interpolate(image, scale_factor=0.5, mode="bilinear")
        image = self.normalize(image)
        
        outputs = []
        final_layer = max(self.feature_layers)
        
        for idx, layer in enumerate(self.vgg.features):
            image = layer(image)
            if idx in self.feature_layers:
                outputs.append(image)
            if idx == final_layer:
                break
                
        return torch.cat(outputs, dim=1).squeeze(dim=0)
    
    
def nnfm_argmin_cos_distance(a, b, center=False):
    """
    a: [c, n],
    b: [c, m]
    """

    # 归一化b（每个m向量单位化）
    b_norm = torch.norm(b, dim=0, keepdim=True)  # [1, m]
    b_normalized = b / (b_norm + 1e-8)          # [c, m]

    z_best = []
    loop_batch_size = int(1e8 // b.shape[1])     # 动态分批次防止内存溢出
    
    # 分批次处理a的hw维度
    for i in range(0, a.shape[1], loop_batch_size):
        a_batch = a[:, i:i+loop_batch_size]      # [c, L]
        
        # 归一化当前批次的a
        a_batch_norm = torch.norm(a_batch, dim=0, keepdim=True)  # [1, L]
        a_batch_normalized = a_batch / (a_batch_norm + 1e-8)     # [c, L]

        # 计算余弦相似度矩阵
        # simlarity : [h*w, k] k为总风格特征数量
        similarity = torch.mm(a_batch_normalized.t(), b_normalized)  # [L, m]
        d_mat = 1.0 - similarity  # 转换为距离
        
        # 找到每个位置的最小距离索引
        z_best_batch = torch.argmin(d_mat, dim=1)  # [L]
        z_best.append(z_best_batch)

    return torch.cat(z_best, dim=0)

def nnfm_feat_replace(A, B, Mat):
    # c, h, w = A.shape
    # A_flat = A.reshape(c, -1)
    # B_flat = B.reshape(c, -1)
    # indices:[h*w]
    indices = nnfm_argmin_cos_distance(A, B)
    C_flat = B[:, indices]
    C_matrix = Mat[:, indices]
    # ic(A.shape)
    # ic(B.shape)
    # ic(C_flat.shape)
    # ic(C_matrix.shape)
    # exit(0)
    return C_flat, C_matrix


def prior_argmin_cos_distance(a, b, Mat, p_mask, p_feats, p_Mat):
    """
    a: [b, c, hw],
    b: [b, c, h2w2]
    """

    # 归一化b（每个m向量单位化）
    b_norm = torch.norm(b, dim=0, keepdim=True)  # [1, m]
    b_normalized = b / (b_norm + 1e-8)          # [c, m]
    
    _, k = b_norm.shape
    
    p_feat_norm = torch.norm(p_feats, dim=0, keepdim=True)
    p_feat_normalized = p_feats / (p_feat_norm + 1e-8)

    z_best = []
    loop_batch_size = int(1e8 // b.shape[1])     # 动态分批次防止内存溢出
    
    # 分批次处理a的hw维度
    for i in range(0, a.shape[1], loop_batch_size):
        a_batch = a[:, i:i+loop_batch_size]      # [c, L]
        p_Mat_batch = p_Mat[:, i:i+loop_batch_size]
        p_feats_batch = p_feats[:, i:i+loop_batch_size]
        p_mask_batch = p_mask[:, i:i+loop_batch_size]
        cal_p_mask = p_mask_batch.view(-1, 1).repeat(1, k)
        
        
        # 归一化当前批次的a
        a_batch_norm = torch.norm(a_batch, dim=0, keepdim=True)  # [1, L]
        a_batch_normalized = a_batch / (a_batch_norm + 1e-8)     # [c, L]
        
        p_feats_batch_norm = torch.norm(p_feats_batch, dim=0, keepdim=True)
        p_feats_batch_normalized = p_feats_batch / (p_feats_batch_norm + 1e-8)

        # 计算余弦相似度矩阵 (nnfm)
        similarity = torch.mm(a_batch_normalized.t(), b_normalized)  # [L, m]
        d_mat = 1.0 - similarity  # 转换为距离
        
        # prior， 有需要调整的超参
        # similarity2 = torch.mm(p_feats_batch_normalized.t(), b_normalized) + torch.mm(p_Mat_batch.t(), Mat)
        # d_mat2 = (1.0 - similarity2) * cal_p_mask
        
        # TODO: 旋转操作
        
        # Mat distance
        target_Mat = p_Mat_batch.view(-1, 1)
        # 有需要调整的超参
        l1_dist = torch.abs(target_Mat - Mat)
        
        # distance聚合
        # d_mat_all = d_mat + d_mat2 + l1_dist
        d_mat_all = d_mat + 100 * l1_dist
        
        
        # 找到每个位置的最小距离索引
        z_best_batch = torch.argmin(d_mat_all, dim=1)  # [L]
        z_best.append(z_best_batch)

    return torch.cat(z_best, dim=0)

def prior_feat_replace(A, B, Mat, p_mask, p_feats, p_Mat):
    # c, h, w = A.shape
    # A_flat = A.reshape(c, -1)
    # B_flat = B.reshape(c, -1)
    
    # Mat_flat = Mat.reshape(1, -1)
    # p_mask_flat = p_mask.reshape(1, -1)
    # p_feats_flat = p_feats.reshape(c, -1)
    # p_Mat_flat = p_Mat.reshape(1, -1)
    
    # indices:[h*w]
    indices = prior_argmin_cos_distance(A, B, Mat, p_mask, p_feats, p_Mat)
    C_flat = B[:, indices]
    C_matrix = Mat[:, indices] * (1 - p_mask) + p_Mat * p_mask
    
    return C_flat, C_matrix
    
    
def content_loss_fn(render_feats_list, scene_feats_list):
    content_loss = 0
    coefficient = 1
    for (render_feat, scene_feat) in zip(render_feats_list, scene_feats_list):
        content_loss += coefficient * torch.mean((render_feat - scene_feat) ** 2)
        coefficient /= 5
    return content_loss