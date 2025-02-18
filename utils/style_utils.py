import torch
import torchvision
from torchvision.models import VGG16_Weights
import torch.nn.functional as F
from sklearn.cluster import KMeans
import numpy as np
import cv2
from PIL import Image
from preprocess import render_RGBcolor_images
import os

def labels_downscale(labels, new_dim):
    """
    Downscales the labels to a new dimension.

    @param labels: Tensor of labels. Shape: [H, W]
    @param new_dim: Tuple of new dimensions (NH, NW)
    @return: Downscaled labels
    """
    H, W = labels.shape
    NH, NW = new_dim
    r_indices = torch.linspace(0, H-1, NH).long()
    c_indices = torch.linspace(0, W-1, NW).long()
    return labels[r_indices[:, None], c_indices]

class SharedStorage:
    target_list = {}
    target_matrix = {}
    style_feats = []
    style_matrix = []

class VGGLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()    
        mean=[0.485, 0.456, 0.406]
        std=[0.229, 0.224, 0.225]
        self.normalize = torchvision.transforms.Normalize(mean=mean, std=std)
        self.vgg = torchvision.models.vgg16(weights=VGG16_Weights.DEFAULT).eval().to("cuda")
        self.layers = [11, 13, 15]
    
    def get_feats(self, image: torch.Tensor):
        """
        Get features from the VGG network.

        @param image: Tensor of the image. Shape: [B, C, H, W]
        @return: Concatenated features from specified layers. 
        """
        image = self.normalize(image)
        final_ix = max(self.layers)
        outputs = []

        for ix, layer in enumerate(self.vgg.features):
            image = layer(image)
            if ix in self.layers:
                outputs.append(image.squeeze())

            if ix == final_ix:
                break

        return torch.cat(outputs)
    
    def forward(self, id, render_image):
        def cos_loss(a, b):
            a_norm = (a * a).sum(1, keepdims=True).sqrt()
            b_norm = (b * b).sum(1, keepdims=True).sqrt()
            a_tmp = a / (a_norm + 1e-8)
            b_tmp = b / (b_norm + 1e-8)
            cossim = (a_tmp * b_tmp).sum(1)
            cos_d = 1.0 - cossim
            return cos_d.mean()
        render_image = render_image.unsqueeze(0)
        render_image = F.interpolate(render_image, scale_factor=0.5, mode="bilinear")
        
        render_feats = self.get_feats(render_image)
        # target_feats: [c, h, w]
        return cos_loss(SharedStorage.target_list[id], render_feats)
        
class PriorLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()    
        mean=[0.485, 0.456, 0.406]
        std=[0.229, 0.224, 0.225]
        self.normalize = torchvision.transforms.Normalize(mean=mean, std=std)
        self.vgg = torchvision.models.vgg16(weights=VGG16_Weights.DEFAULT).eval().to("cuda")
        self.layers = [11, 13, 15]
    
    def get_feats(self, image: torch.Tensor):
        """
        Get features from the VGG network.

        @param image: Tensor of the image. Shape: [B, C, H, W]
        @return: Concatenated features from specified layers. 
        """
        image = self.normalize(image)
        final_ix = max(self.layers)
        outputs = []

        for ix, layer in enumerate(self.vgg.features):
            image = layer(image)
            if ix in self.layers:
                outputs.append(image.squeeze())

            if ix == final_ix:
                break

        return torch.cat(outputs)
    
    def argmin_cos_distance(self, a, b, Mat, p_mask, p_feats, p_Mat):
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
            similarity2 = torch.mm(p_feats_batch_normalized.t(), b_normalized) + torch.mm(p_Mat_batch.t(), Mat)
            d_mat2 = (1.0 - similarity2) * cal_p_mask
            
            # Mat distance
            target_Mat = p_Mat_batch.view(-1, 1)
            # 有需要调整的超参
            l1_dist = torch.abs(target_Mat - Mat)
            
            # distance聚合
            # d_mat_all = d_mat + d_mat2 + l1_dist
            d_mat_all = d_mat + l1_dist
            
            # 找到每个位置的最小距离索引
            z_best_batch = torch.argmin(d_mat_all, dim=1)  # [L]
            z_best.append(z_best_batch)

        return torch.cat(z_best, dim=0)
    
    def nn_feat_replace(self, A, B, Mat, p_mask, p_feats, p_Mat):
        c, h, w = A.shape
        A_flat = A.reshape(c, -1)
        B_flat = B.reshape(c, -1)
        
        Mat_flat = Mat.reshape(1, -1)
        p_mask_flat = p_mask.reshape(1, -1)
        p_feats_flat = p_feats.reshape(c, -1)
        p_Mat_flat = p_Mat.reshape(1, -1)
        
        # indices:[h*w]
        indices = self.argmin_cos_distance(A_flat, B_flat, Mat_flat, p_mask_flat, p_feats_flat, p_Mat_flat)
        C_flat = B[:, indices]
        C_matrix = Mat[:, indices]
        return C_flat.reshape(c, h, w), C_matrix.reshape(1, h, w)
    
    def forward(self, render_image, mask, pos_pre, id):
        
        def cos_loss(a, b):
            a_norm = (a * a).sum(1, keepdims=True).sqrt()
            b_norm = (b * b).sum(1, keepdims=True).sqrt()
            a_tmp = a / (a_norm + 1e-8)
            b_tmp = b / (b_norm + 1e-8)
            cossim = (a_tmp * b_tmp).sum(1)
            cos_d = 1.0 - cossim
            return cos_d.mean()
        
        
        # 读入先验数据
        # prior_target和prior_matrix还需要进行双线性插值进行转换才可以使用
        prior_target = SharedStorage.target_list[id - 1]
        prior_matrix = SharedStorage.target_matrix[id - 1]
        
        h,w = mask.shape
        rows = torch.arange(h, device=prior_target.device)  # (h,)
        cols = torch.arange(w, device=prior_target.device)  # (w,)
        grid_i, grid_j = torch.meshgrid(rows, cols, indexing="ij")
        pos = torch.stack([grid_i, grid_j], dim=-1)
        
        # 处理当前视图
        render_image = render_image.unsqueeze(0)
        render_image = F.interpolate(render_image, scale_factor=0.5, mode="bilinear")
        
        # (c, h, w)
        render_feats = self.get_feats(render_image)
        
        # 处理先验mask和对应问题
        _, fh, fw = render_feats.shape
        fh_grid, fw_grid = torch.meshgrid(
            torch.arange(fh, device=prior_target.device),
            torch.arange(fw, device=prior_target.device),
            indexing='ij'
        )
        img_h = fh_grid * 8 + 4
        img_w = fw_grid * 8 + 4
        prior_idx = torch.zeros(fh, fw, 2).to(prior_target.device)
        prior_idx = pos[img_h, img_w]
        rows = prior_idx[:, :, 0]
        cols = prior_idx[:, :, 1]
        prior_mask = mask[rows, cols]
        
        prior_idx = (pos_pre[img_h, img_w] - 4.0) / 8.0
        # 转换成
        grid = prior_idx.unsqueeze(0)
        
        prior_feats = result = F.grid_sample(prior_target.unsqueeze(0), grid, mode='bilinear', align_corners=False)
        prior_feats = prior_feats.squeeze()
        # mask_feats = mask[]
        
        
        # 计算新的target_feats
        # 其实就是找一个index, 这个index的特征和原图，先验特征和先验矩阵的总相似情况的argmin
        # 需要注意的是mask，即有先验才用先验，没有先验则和原来nnfm一样（除了if else如何有好的实现？， 应该是要修改求argmin的时候先验项的系数，mask为0时置0）
        target_feats = torch.zeros_like(render_feats)
        
        with torch.no_grad():
            target_feats, target_matrix = self.nn_feat_replace(render_feats, SharedStorage.style_feats, SharedStorage.style_matrix, prior_mask, prior_feats, prior_matrix)
        
        
        # 更新target_feats
        SharedStorage.target_list[id] = target_feats
        SharedStorage.target_matrix[id] = target_matrix
        # 算loss
        # target_feats: [c, h, w]
        return cos_loss(target_feats, render_feats)

class StyleLoss(torch.nn.Module):
    def __init__(self, pre, scene, override_matches=None):
        """
        Initialize VGG and style features.

        @param pre: PreProcess instance
        @param override_matches: Overrides for matches if any
        """
        super().__init__()        
        
        self.viewpoint_stack = scene.getTrainCameras()
        self.scene_classes = pre.scene_classes
        self.style_classes = pre.style_classes
        self.device = pre.device
        
        # self.style_masks = pre.style_masks
        self.scene_masks = pre.scene_masks
        
        self.matches = pre.matches
        
        mean=[0.485, 0.456, 0.406]
        std=[0.229, 0.224, 0.225]
        self.normalize = torchvision.transforms.Normalize(mean=mean, std=std)
        
        self.vgg = torchvision.models.vgg16(weights=VGG16_Weights.DEFAULT).eval().to("cuda")
        
        self.layers = [11, 13, 15]
        
        # 确定特征图尺寸
        with torch.no_grad():
            gt_image = self.viewpoint_stack[0].original_image.cuda()
            gt_image = gt_image.unsqueeze(0)
            gt_image = F.interpolate(gt_image, scale_factor=0.5, mode="bilinear")
            gt_feats = self.get_feats(gt_image)
        fc, fh, fw = gt_feats.shape
        
        with torch.no_grad():
            self.style_feats = []
            self.style_matrix = []
            
            
            for _, style_image in enumerate(pre.style_image_list):
                # 纹理图像增强
                style_img_width = style_image.shape[2]
                style_img_height = style_image.shape[1]
                for i in range(0, 4):
                    # 先假设错切参数为0，旋转角度由i得到
                    Hx = Hy = 0
                    theta = i * 90.0
                    
                    # 获得线性变换矩阵（包括旋转角度和错切参数）
                    M, M_parameter = self.generate_transformation_matrix(theta, Hx, Hy, style_img_width, style_img_height)
                    
                    # 根据i获得增强后的图片
                    # new_image = F.affine(style_image, theta, [0,9], 1.0, [Hx, Hy], resample=Image.BICUBIC)
                    new_image = self.tensor_img_transformation(style_image, M, i)
                    
                    # extract vgg feature
                    # self.style_feats.append(self.get_feats(style_image.unsqueeze(0)))
                    img_feats = self.get_feats(new_image.unsqueeze(0))
                    c, h, w = img_feats.shape
                    img_feats = img_feats.view(c, -1)
                    _, num_clusters = img_feats.shape
                    
                    # # 处理为k-means可用形式
                    # c, h, w = img_feats.shape
                    # img_feats_flat = img_feats.permute(1, 2, 0).reshape(-1, c)
                    # img_feats_np = img_feats_flat.cpu().numpy()
                    
                    # # k-means
                    # # TODO: 超参调整
                    # num_clusters = 40
                    # kmeans = KMeans(n_clusters=num_clusters, random_state=42, max_iter=300)
                    # kmeans.fit(img_feats_np)
                    
                    # # 获取每个像素对应的聚类标签
                    # cluster_labels = kmeans.labels_  # [h * w]

                    # # 获取聚类中心
                    # cluster_centers = kmeans.cluster_centers_  # [num_clusters, c]

                    # # 将每个像素替换为其所属聚类的中心向量
                    # discretized_feats_flat = torch.tensor(cluster_centers, device=img_feats.device)

                    # # 将特征还原回原始形状 [c, -1]
                    # img_feats = discretized_feats_flat.view(-1, c).permute(1, 0)
                    
                    # 将新特征组加入总特征集合（当前未考虑深度分组）
                    
                    self.style_feats.append(img_feats)
                    
                    # M_tensor = torch.from_numpy(M_parameter)
                    # 目前先只存了旋转角度，用于求loss，而不是
                    M_tensor = torch.tensor(theta)
                    matrix_list = torch.stack([M_tensor] * num_clusters).to("cuda").unsqueeze(0)
                    self.style_matrix.append(matrix_list)
                    if theta == 0:
                        A = img_feats
                        A_mat = matrix_list
                    if theta == 90:
                        B = img_feats
                        B_mat = matrix_list
            # 造第一个视图的feature map用于调试
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
            SharedStorage.target_list[0] = C
            SharedStorage.target_matrix[0] = C_mat
            
            
            self.style_feats = torch.cat(self.style_feats, dim=1)
            self.style_matrix = torch.cat(self.style_matrix, dim=1)
            
            SharedStorage.style_feats = self.style_feats
            SharedStorage.style_matrix = self.style_matrix
            
            
            # print(self.style_feats.shape)
            # print("!!!")
            
            # style_feats = self.style_feats  # [c, h, w]
            # c, h, w = style_feats.shape
            # style_feats_flat = style_feats.permute(1, 2, 0).reshape(-1, c)
            # # print(style_feats_flat.shape)
            # style_feats_np = style_feats_flat.cpu().numpy()
            # num_clusters = 64
            # # 使用 KMeans 进行聚类
            # kmeans = KMeans(n_clusters=num_clusters, random_state=42, max_iter=300)
            # kmeans.fit(style_feats_np)

            # # 获取每个像素对应的聚类标签
            # cluster_labels = kmeans.labels_  # [h * w]

            # # 获取聚类中心
            # cluster_centers = kmeans.cluster_centers_  # [num_clusters, c]

            # # 将每个像素替换为其所属聚类的中心向量
            # discretized_feats_flat = torch.tensor(cluster_centers, device=style_feats.device)

            # 将特征还原回原始形状 [c, h, w]
            # self.style_feats = discretized_feats_flat.view(-1, c).permute(1, 0)
            # print(self.style_feats.shape)
                
                
            # self.style_masks = []
            # for i, mask in enumerate(pre.style_masks):
            #     self.style_masks.append(labels_downscale(mask, self.style_feats[i].shape[-2:]))                 
            # self.style_feats = self.style_feats.view(self.style_feats.shape[0], -1)
            # self.style_masks = torch.cat(self.style_masks, dim=1)
    
    
        
    def generate_transformation_matrix(self, angle, shear_x, shear_y, image_width, image_height):
        """
        Generate the linear transformation matrix for a given rotation angle, shear parameters,
        and image dimensions (to rotate around the image center).
        
        @param angle: Rotation angle in degrees.
        @param shear_x: Shear factor along the x-axis.
        @param shear_y: Shear factor along the y-axis.
        @param image_width: The width of the image.
        @param image_height: The height of the image.
        @return: The combined transformation matrix (3x3).
        """
        # Step 1: Rotation matrix (around the center)
        angle_rad = np.radians(angle)
        rotation_matrix = np.array([
            [np.cos(angle_rad), -np.sin(angle_rad), 0],
            [np.sin(angle_rad), np.cos(angle_rad), 0],
            [0, 0, 1]
        ])
        
        # Step 2: Shear matrix
        shear_matrix = np.array([
            [1, shear_x, 0],
            [shear_y, 1, 0],
            [0, 0, 1]
        ])
        
        # Step 3: Combine rotation and shear matrices
        combined_matrix = np.dot(shear_matrix, rotation_matrix)
        
        # Step 4: Translate image center to origin, apply rotation, then translate back
        center_x = image_width / 2
        center_y = image_height / 2
        
        # Translate to origin, rotate, then translate back
        translation_matrix_to_origin = np.array([
            [1, 0, -center_x],
            [0, 1, -center_y],
            [0, 0, 1]
        ])
        
        translation_matrix_back = np.array([
            [1, 0, center_x],
            [0, 1, center_y],
            [0, 0, 1]
        ])
        
        # Combine all the transformations
        final_matrix = np.dot(translation_matrix_back, np.dot(combined_matrix, translation_matrix_to_origin))
        
        return final_matrix, combined_matrix
    
    def center_crop(self, image, crop_size):
        """
        对图像执行中心裁剪。
        
        @param image: 输入图像 (NumPy array)。
        @param crop_size: 裁剪尺寸 (width, height)。
        @return: 裁剪后的图像 (NumPy array)。
        """
        # 获取图像尺寸
        h, w = image.shape[:2]
        
        # 计算裁剪起始点
        start_x = w//2 - crop_size[0]//2
        start_y = h//2 - crop_size[1]//2
        
        # 执行裁剪
        cropped_image = image[start_y:start_y + crop_size[1], start_x:start_x + crop_size[0]]
        
        cropped_image = cv2.resize(cropped_image, (256, 256), interpolation=cv2.INTER_LANCZOS4)
        return cropped_image
    
    def tensor_img_transformation(self, tensor_image, transformation_matrix, idx):
        # Convert tensor to NumPy array (H, W, C)
        image_numpy = tensor_image.permute(1, 2, 0).cpu().numpy()
        
        # Get image size
        rows, cols, _ = image_numpy.shape
        
        # Apply the affine transformation using the top 2x3 part of the transformation matrix
        affine_matrix = transformation_matrix[:2, :]
        transformed_image = cv2.warpAffine(image_numpy, affine_matrix, (cols, rows), flags=cv2.INTER_LANCZOS4)
        transformed_image = self.center_crop(transformed_image, (int(cols / 1.3), int(rows / 1.3)))
        
        # Convert back to Tensor and normalize
        transformed_tensor = torch.from_numpy(transformed_image).permute(2, 0, 1).float().to(self.device)
        
        check_path = "/home/lwj/data/TAT-GS/check/"
        render_RGBcolor_images(os.path.join(check_path, f"{int(idx):04d}.png"), transformed_tensor)
        
        return transformed_tensor
    
    def get_matches(self):
        """
        Get matches for style and scene classes if no override matches.

        @return: List of matches
        """
        matches = []
        for i in range(self.scene_classes):
            matches.append(i % self.style_classes)
            
        return matches
    
    def get_feats(self, image: torch.Tensor):
        """
        Get features from the VGG network.

        @param image: Tensor of the image. Shape: [B, C, H, W]
        @return: Concatenated features from specified layers. 
        """
        image = self.normalize(image)
        final_ix = max(self.layers)
        outputs = []

        for ix, layer in enumerate(self.vgg.features):
            image = layer(image)
            if ix in self.layers:
                outputs.append(image.squeeze())

            if ix == final_ix:
                break

        return torch.cat(outputs)
    
    def calc_image_tv_loss(self, render_image):
        """
        Calculate image total variation loss.

        @param render_image: Tensor of the render image. Shape: [B, C, H, W]
        @return: Image total variation loss
        """
        w_variance = torch.mean(torch.pow(render_image[:, :, :-1] - render_image[:, :, 1:], 2))
        h_variance = torch.mean(torch.pow(render_image[:, :-1, :] - render_image[:, 1:, :], 2))
        img_tv_loss = (h_variance + w_variance) / 2.0
        return img_tv_loss
    
    def calc_style_loss(self, scene_mask, render_feats, id):
        """
        Calculate style loss (to be overridden by subclasses).

        @param scene_mask: Tensor of scene masks. Shape: [H, W]
        @param render_feats: Tensor of render features.
        @return: Style loss
        """
        raise NotImplementedError()
    
    def forward(self, scene_mask, gt_image, render_image, id):
        """
        Forward pass to calculate losses.

        @param scene_mask: Tensor of scene masks. Shape: [H, W]
        @param gt_image: Tensor of ground truth image. Shape: [C, H, W]
        @param render_image: Tensor of render image. Shape: [C, H, W]
        @return: Tuple of style loss, content loss, and image total variation loss
        """
        # feats are all [768, new_h, new_w] in this paper.
        render_image = render_image.unsqueeze(0)
        render_image = F.interpolate(render_image, scale_factor=0.5, mode="bilinear")
        gt_image = gt_image.unsqueeze(0)
        gt_image = F.interpolate(gt_image, scale_factor=0.5, mode="bilinear")
        
        render_feats = self.get_feats(render_image)
        with torch.no_grad():
            gt_feats = self.get_feats(gt_image)
            scene_mask = labels_downscale(scene_mask, render_feats.shape[-2:])
        
        style_loss = self.calc_style_loss(scene_mask, render_feats, id)
        img_tv_loss = self.calc_image_tv_loss(render_image)
        content_loss = torch.mean((render_feats - gt_feats) ** 2)
        
        return (style_loss, content_loss, img_tv_loss)
    
    
class FASTLoss(StyleLoss):
        
    def cal_p(self, cf, sf, mask=None):
        """
        Calculate the transformation matrix.

        @param cf: Tensor of content features. Shape: [C * kernel_size, H * W]
        @param sf: Tensor of style features. Shape: [C * kernel_size, H * W]
        @param mask: Optional mask tensor. Shape: [H * W, H * W]
        @return: Transformation matrix P. Shape: [C, C]
        """
        cf_size = cf.size()
        sf_size = sf.size()
        
        k_cross = 5
        # k_cross = self.cfg.MAST_CORE.K_CROSS

        cf_temp = cf
        sf_temp = sf

        # if self.cfg.MAST_CORE.MAX_USE_NUM == -1:
        # ########################################
        # normalize
        cf_n = F.normalize(cf, 2, 0)
        sf_n = F.normalize(sf, 2, 0)
        # #########################################

        dist = torch.mm(cf_n.t(), sf_n)  # inner product,the larger the value, the more similar
        if mask is not None:
            mask = mask.type_as(dist).to(self.device)
            dist = torch.mul(dist, mask)

        hcwc, hsws = cf_size[1], sf_size[1]
        U = torch.zeros(hcwc, hsws).type_as(cf_n).to(self.device)  # construct affinity matrix "(h*w)*(h*w)"

        index = torch.topk(dist, k_cross, 0)[1]  # find indices k nearest neighbors along row dimension
        value = torch.ones(k_cross, hsws).type_as(cf_n).to(self.device) # "KCross*(h*w)"
        U.scatter_(0, index, value)  # set weight matrix

        index = torch.topk(dist, k_cross, 1)[1]  # find indices k nearest neighbors along col dimension
        value = torch.ones(hcwc, k_cross).type_as(cf_n).to(self.device)
        U.scatter_(1, index, value)  # set weight matrix
        
        n_cs = torch.sum(U)
        U = U / n_cs
        D1 = torch.diag(torch.sum(U, dim=1)).type_as(cf).to(self.device)
        
        A = torch.mm(torch.mm(cf_temp, D1), cf_temp.t())
        regularization_term = torch.eye(A.size()[0]).type_as(A).to(self.device) * 1e-12
        A += regularization_term
        B = torch.mm(torch.mm(cf_temp, U), sf_temp.t())
        
        try:
            p = torch.linalg.solve(A, B)
        except Exception as e:
            print(e)
            p = torch.eye(cf_size[0]).type_as(cf).to(self.device)
        return p


    def transform(self, render_feats, style_feats):
        """
        Transform render features to match style features.

        @param render_feats: Tensor of render features. Shape: [N, C, H, W]
        @param style_feats: Tensor of style features.
        @return: Transformed render features
        """
        # print(render_feats.shape, style_feats.shape)
        p = self.cal_p(render_feats, style_feats, None)
        return torch.mm(p.t(), render_feats).unsqueeze(0)
    
    def calc_style_loss(self, scene_mask, render_feats):
        """
        Calculate FAST loss.

        @param scene_mask: Tensor of scene masks. Shape: [H, W]
        @param render_feats: Tensor of render features.
        @return: FAST style loss
        """
        
        def cos_loss(a, b):
            a_norm = (a * a).sum(1, keepdims=True).sqrt()
            b_norm = (b * b).sum(1, keepdims=True).sqrt()
            a_tmp = a / (a_norm + 1e-8)
            b_tmp = b / (b_norm + 1e-8)
            cossim = (a_tmp * b_tmp).sum(1)
            cos_d = 1.0 - cossim
            return cos_d.mean()
        
        target_feats = torch.zeros_like(render_feats)
        with torch.no_grad():
            for i in range(self.scene_classes):
                render_idx = (scene_mask == i)
                # style_idx = (self.style_masks == self.matches[i])
                
                target_feats[:, render_idx] = self.transform(
                    render_feats[:, render_idx],
                    self.style_feats.view(self.style_feats.shape[0], -1)[:])
                
        return cos_loss(target_feats, render_feats)
    
    
    
class NNFMLoss(StyleLoss):
    
    
    def argmin_cos_distance(self, a, b, center=False):
        """
        a: [b, c, hw],
        b: [b, c, h2w2]
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
    
    def nn_feat_replace(self, A, B, Mat):
        c, h, w = A.shape
        A_flat = A.reshape(c, -1)
        B_flat = B.reshape(c, -1)
        # indices:[h*w]
        indices = self.argmin_cos_distance(A_flat, B_flat)
        C_flat = B[:, indices]
        C_matrix = Mat[:, indices]
        return C_flat.reshape(c, h, w), C_matrix.reshape(1, h, w)
    
    def calc_style_loss(self, scene_mask, render_feats, id):   
        """
        calculate NNFM loss
        """
        
        def cos_loss(a, b):
            a_norm = (a * a).sum(1, keepdims=True).sqrt()
            b_norm = (b * b).sum(1, keepdims=True).sqrt()
            a_tmp = a / (a_norm + 1e-8)
            b_tmp = b / (b_norm + 1e-8)
            cossim = (a_tmp * b_tmp).sum(1)
            cos_d = 1.0 - cossim
            return cos_d.mean()
        
        def cosine_dists(a, b):
            if len(b.shape) == 3:
                c, h, w = a.shape
                
                a = a.permute(1, 2, 0).view(-1, c)
                b = b.permute(1, 2, 0).view(-1, c)
            elif len(b.shape) == 2:
                c, hw = b.shape
                a = a.permute(1, 2, 0).view(-1, c)
                b = b.permute(1, 0).view(-1, c)
                
            a_norm = (a * a).sum(1, keepdims=True).sqrt()
            b_norm = (b * b).sum(1, keepdims=True).sqrt()
            a_tmp = a / (a_norm + 1e-8)
            b_tmp = b / (b_norm + 1e-8)
            return 1.0 - torch.matmul(a_tmp, b_tmp.T)
        
        # dists = cosine_dists(render_feats, self.style_feats)
        # # kmin_dist, ind = torch.topk(dists, k=5, dim=1)
        # # kmean_dist = torch.mean(kmin_dist, dim=1)
        # # print(kmean_dist.shape)
        # # style_knnfm_loss = torch.mean(kmean_dist)
        # min_dist = torch.amin(dists, dim=1)
        # # print(min_dist.shape)
        # style_loss = torch.mean(min_dist)
        # return style_loss
        
        if id == 0:
            target_feats = SharedStorage.target_list[0]
            target_matrix = SharedStorage.target_matrix[0]
        else:
            target_feats = torch.zeros_like(render_feats)
            
            with torch.no_grad():
                target_feats, target_matrix = self.nn_feat_replace(render_feats, self.style_feats, self.style_matrix)
                # for i in range(self.scene_classes):
                #     render_idx = (scene_mask == i)
                #     style_idx = (self.style_masks == self.matches[i])
                    
                #     target_feats[:, render_idx] = self.nn_feat_replace(
                #         render_feats[:, render_idx],
                #         self.style_feats[:, style_idx])
        
        SharedStorage.target_list[id] = target_feats
        SharedStorage.target_matrix[id] = target_matrix
        return cos_loss(target_feats, render_feats)
    
class KNNFMLoss(StyleLoss):
    
    # def nn_feat_replace(a, b):
    #     n, c, h, w = a.size()
    #     n2, c, h2, w2 = b.size()

    #     assert (n == 1) and (n2 == 1)
        
    #     c, pixel = a.size()

    #     a_flat = a.view(n, c, -1)
    #     b_flat = b.view(n2, c, -1)
    #     b_ref = b_flat.clone()

    #     z_new = []
        
        
    #     z_best = argmin_cos_distance(a_flat[i : i + 1], b_flat[i : i + 1])
    #     z_best = z_best.unsqueeze(1).repeat(1, c, 1)
    #     feat = torch.gather(b_ref, 2, z_best)
    #     z_new.append(feat)
        
    #     return z_new
    
    def calc_style_loss(self, scene_mask, render_feats):   
        """
        calculate NNFM loss
        """
        
        def cos_loss(a, b):
            a_norm = (a * a).sum(1, keepdims=True).sqrt()
            b_norm = (b * b).sum(1, keepdims=True).sqrt()
            a_tmp = a / (a_norm + 1e-8)
            b_tmp = b / (b_norm + 1e-8)
            cossim = (a_tmp * b_tmp).sum(1)
            cos_d = 1.0 - cossim
            return cos_d.mean()
        
        def cosine_dists(a, b):
            if len(a.shape) == 3:
                c, h, w = a.shape
                
                a = a.permute(1, 2, 0).view(-1, c)
                b = b.permute(1, 2, 0).view(-1, c)
            a_norm = (a * a).sum(1, keepdims=True).sqrt()
            b_norm = (b * b).sum(1, keepdims=True).sqrt()
            a_tmp = a / (a_norm + 1e-8)
            b_tmp = b / (b_norm + 1e-8)
            return 1.0 - torch.matmul(a_tmp, b_tmp.T)
        
        
        dists = -1.0 * cosine_dists(render_feats, self.style_feats)
        kmin_dist, ind = torch.topk(dists, k=5, dim=1)
        kmean_dist = torch.mean(kmin_dist, dim=1)
        # print(kmean_dist.shape)
        style_knnfm_loss = -1.0 * torch.mean(kmean_dist)
        return style_knnfm_loss
    

class GRAMLoss(StyleLoss):
        
    def calc_style_loss(self, scene_mask, render_feats):
        """
        calculate GRAM loss
        """
        def gram_matrix(feature_maps, center=False):
            """
            feature_maps: b, c, h, w
            gram_matrix: b, c, c
            """
            c, h, w = feature_maps.size()
            features = feature_maps.view(c, h * w)
            if center:
                features = features - features.mean(dim=-1, keepdims=True)
            G = torch.mm(features, features.t())
            return G
        
        style_loss = torch.mean((gram_matrix(render_feats) - gram_matrix(self.style_feats)) ** 2)
        return style_loss