import torch
import torchvision
from torchvision.models import VGG16_Weights
import torch.nn.functional as F

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

class StyleLoss(torch.nn.Module):
    def __init__(self, pre, override_matches=None):
        """
        Initialize VGG and style features.

        @param pre: PreProcess instance
        @param override_matches: Overrides for matches if any
        """
        super().__init__()        
        
        self.scene_classes = pre.scene_classes
        self.style_classes = pre.style_classes
        self.device = pre.device
        
        self.style_masks = pre.style_masks
        self.scene_masks = pre.scene_masks
        
        self.matches = pre.matches
        
        mean=[0.485, 0.456, 0.406]
        std=[0.229, 0.224, 0.225]
        self.normalize = torchvision.transforms.Normalize(mean=mean, std=std)
        
        self.vgg = torchvision.models.vgg16(weights=VGG16_Weights.DEFAULT).eval().to("cuda")
        
        self.layers = [11, 13, 15]
        with torch.no_grad():
            self.style_feats = []
            for _, style_image in enumerate(pre.style_image_list):
                self.style_feats.append(self.get_feats(style_image.unsqueeze(0)))
                print(self.style_feats[_].shape)
            
            self.style_masks = []
            for i, mask in enumerate(pre.style_masks):
                self.style_masks.append(labels_downscale(mask, self.style_feats[i].shape[-2:]))
                
            
            self.style_feats = torch.cat(self.style_feats, dim=2)
            self.style_masks = torch.cat(self.style_masks, dim=1)
        
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
    
    def calc_style_loss(self, scene_mask, render_feats):
        """
        Calculate style loss (to be overridden by subclasses).

        @param scene_mask: Tensor of scene masks. Shape: [H, W]
        @param render_feats: Tensor of render features.
        @return: Style loss
        """
        raise NotImplementedError()
    
    def forward(self, scene_mask, gt_image, render_image):
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
        
        style_loss = self.calc_style_loss(scene_mask, render_feats)
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
                style_idx = (self.style_masks == self.matches[i])
                
                target_feats[:, render_idx] = self.transform(
                    render_feats[:, render_idx],
                    self.style_feats[:, style_idx])
                
        return cos_loss(target_feats, render_feats)
    
    
    
class NNFMLoss(StyleLoss):
    
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
        
        
        dists = cosine_dists(render_feats, self.style_feats)
        # kmin_dist, ind = torch.topk(dists, k=5, dim=1)
        # kmean_dist = torch.mean(kmin_dist, dim=1)
        # print(kmean_dist.shape)
        # style_knnfm_loss = torch.mean(kmean_dist)
        min_dist = torch.amin(dists, dim=1)
        # print(min_dist.shape)
        style_loss = torch.mean(min_dist)
        return style_loss
        
        
        # target_feats = torch.zeros_like(render_feats)
        # with torch.no_grad():
        #     for i in range(self.scene_classes):
        #         render_idx = (scene_mask == i)
        #         style_idx = (self.style_masks == self.matches[i])
                
        #         target_feats[:, render_idx] = self.nn_feat_replace(
        #             render_feats[:, render_idx],
        #             self.style_feats[:, style_idx])
        
                
        # return cos_loss(target_feats, render_feats)
    
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