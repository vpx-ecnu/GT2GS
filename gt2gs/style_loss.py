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
    b_norm = torch.norm(b, dim=0, keepdim=True)  # [1, m]
    b_normalized = b / (b_norm + 1e-8)          # [c, m]

    z_best = []
    loop_batch_size = int(1e8 // b.shape[1])
    
    for i in range(0, a.shape[1], loop_batch_size):
        a_batch = a[:, i:i+loop_batch_size]      # [c, L]
        a_batch_norm = torch.norm(a_batch, dim=0, keepdim=True)  # [1, L]
        a_batch_normalized = a_batch / (a_batch_norm + 1e-8)     # [c, L]
        similarity = torch.mm(a_batch_normalized.t(), b_normalized)  # [L, m]
        d_mat = 1.0 - similarity
        z_best_batch = torch.argmin(d_mat, dim=1)  # [L]
        z_best.append(z_best_batch)

    return torch.cat(z_best, dim=0)

def nnfm_feat_replace(A, B, Mat):
    indices = nnfm_argmin_cos_distance(A, B)
    C_flat = B[:, indices]
    C_matrix = Mat[:, indices]
    return C_flat, C_matrix


def prior_argmin_cos_distance(a, b, Mat, p_mask, p_feats, p_Mat):
    """
    a: [b, c, hw],
    b: [b, c, h2w2]
    """
    b_norm = torch.norm(b, dim=0, keepdim=True)  # [1, m]
    b_normalized = b / (b_norm + 1e-8)          # [c, m]
    
    _, k = b_norm.shape
    
    p_feat_norm = torch.norm(p_feats, dim=0, keepdim=True)
    p_feat_normalized = p_feats / (p_feat_norm + 1e-8)

    z_best = []
    loop_batch_size = int(1e8 // b.shape[1])
    for i in range(0, a.shape[1], loop_batch_size):
        a_batch = a[:, i:i+loop_batch_size]      # [c, L]
        p_Mat_batch = p_Mat[:, i:i+loop_batch_size]
        p_feats_batch = p_feats[:, i:i+loop_batch_size]
        p_mask_batch = p_mask[:, i:i+loop_batch_size]
        cal_p_mask = p_mask_batch.view(-1, 1).repeat(1, k)
        
        a_batch_norm = torch.norm(a_batch, dim=0, keepdim=True)  # [1, L]
        a_batch_normalized = a_batch / (a_batch_norm + 1e-8)     # [c, L]
        
        p_feats_batch_norm = torch.norm(p_feats_batch, dim=0, keepdim=True)
        p_feats_batch_normalized = p_feats_batch / (p_feats_batch_norm + 1e-8)

        similarity = torch.mm(a_batch_normalized.t(), b_normalized)  # [L, m]
        d_mat = 1.0 - similarity 
        
        target_Mat = p_Mat_batch.view(-1, 1)
        l1_dist = torch.abs(target_Mat - Mat)
        d_mat_all = d_mat + 100 * l1_dist
        
        
        z_best_batch = torch.argmin(d_mat_all, dim=1)  # [L]
        z_best.append(z_best_batch)

    return torch.cat(z_best, dim=0)

def prior_feat_replace(A, B, Mat, p_mask, p_feats, p_Mat, flag=True):
    indices = prior_argmin_cos_distance(A, B, Mat, p_mask, p_feats, p_Mat)
    C_flat = B[:, indices]
    C_matrix = Mat[:, indices] * (1 - p_mask) + p_Mat * p_mask


    if flag:
        indices_nnfm = nnfm_argmin_cos_distance(A, B)
        C_matrix_nnfm = Mat[:, indices_nnfm]
        diff = torch.abs(C_matrix - C_matrix_nnfm) % 180.0
        diff = 90.0 - torch.minimum(diff, 180.0 - diff)
    else:
        diff = torch.zeros_like(C_matrix)

    
    return C_flat, C_matrix, diff
    
    
def content_loss_fn(render_feats_list, scene_feats_list, x):
    content_loss = 0
    coefficient = 1
    x_flat = x.view(1, -1)
    for (render_feat, scene_feat) in zip(render_feats_list, scene_feats_list):
        content_loss += coefficient * torch.mean((render_feat - scene_feat) ** 2)
        coefficient /= 1.5
    return content_loss