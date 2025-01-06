import cv2
import torch
import numpy as np
from gaussian_renderer import render

from lang_sam import LangSAM
from tqdm import tqdm
from torchvision.transforms import ToPILImage
import os

def get_matches(scene_classes, style_classes):
    """
    Get matches for style and scene classes if no override matches.

    @return: List of matches
    """
    matches = []
    for i in range(scene_classes):
        matches.append(i % style_classes)
        
    return matches

def render_depth_or_mask_images(path, image):
    """
    Renders and saves depth or mask images.
    
    @param path: The path to save the image.
    @param image: The tensor image to render. Shape: [1, H, W]
    """
    image = image.detach().cpu().numpy().squeeze()
    depth_map_normalized = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
    depth_map_normalized = np.uint8(depth_map_normalized)
    # depth_map_colored = cv2.applyColorMap(depth_map_normalized, cv2.COLORMAP_JET)
    cv2.imwrite(path, depth_map_normalized)
    
def render_RGBcolor_images(path, image):
    """
    Renders and saves RGB color images.
    
    @param path: The path to save the image.
    @param image: The tensor image to render. Shape: [3, H, W]
    """
    image = image.detach().permute(1, 2, 0).clamp(min=0.0, max=1.0).cpu().numpy()
    image = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, image)

def split_prompt(prompt):
    """
    Splits a prompt string into a list of labels.
    
    @param prompt: The string prompt to split.
    @return: List of labels.
    """
    labels = []
    lines = prompt.split(",")
    for line in lines:
        label = line
        labels.append(label)
    return labels
    

class PreProcess:
    """
    Pre-process class for handling and segmenting style and scene images.

    Attributes:
        device: Device to run computations on (CPU/GPU).
        scene: Scene object.
        pipe: Render pipeline.
        bg: Background information.
        viewpoint_stack: List of viewpoints from the scene.
        model: Lang-Seg Model Object.
        style_images: Tensor of the concatenated style images. Shape [C, H, W].
        style_image_list: List of the style images.
        scene_weights: Tensor of results of scene images' masks. Shape [N, C, H, W].
        scene_masks: Tensor of scene masks. Shape: [N, H, W].
        style_masks: Tensor of style masks. Shape: [H, W].
        gaussian_masks: Tensor of Gaussian masks. Shape: [N].
        scene_classes: Number of scene classes.
        style_classes: Number of style classes.
    """

    def __init__(self, scene, style_images_path, scene_prompt, style_prompt, 
                 pipe, bg, device, method, 
                 erode=True, isolate=True, color_transfer=True,
                 override_matches=None) -> None:
        """
        Initializes the pre-process class, loads, and segments style and scene images.
        
        @param scene: Scene object.
        @param style_images_path: List of paths to the style images.
        @param scene_prompt: String of the scene prompt.
        @param style_prompt: String of the style prompt.
        @param pipe: Render pipeline.
        @param bg: Background information.
        @param device: Device to run computations on (CPU/GPU).
        @param method: stylize method
        """
        self.device = device
        self.scene = scene
        
        
        
        self.scene.style_path = ""
        # TODO: change path to dir_path
        # for i, p in enumerate(style_images_path):
        #     cur_path, _ = os.path.splitext(os.path.basename(p))
        #     self.scene.style_path += cur_path
        # self.scene.style_path += method
            
        self.pipe = pipe
        self.bg = bg
        self.viewpoint_stack = scene.getTrainCameras()
        self.model = None
        
        self.unify_scene_images_size()
        self.process_depth_images(scene.gaussians)
        self.process_style_images(style_images_path, style_prompt)
        
        self.scene_weights = self.get_scene_weights(scene_prompt)
        self.scene_masks = self.get_scene_masks(self.scene_weights)
        for i, view in enumerate(self.viewpoint_stack):
            view.scene_mask = self.scene_masks[i]
            
        self.style_masks = self.get_style_masks(style_prompt)
        
        self.scene_classes = int(torch.max(self.scene_masks)) + 1
        
        self.style_classes = -1
        for i, mask in enumerate(self.style_masks):
            self.style_classes = max(self.style_classes, int(torch.max(mask)) + 1)
            
        if override_matches is None:
            self.matches = get_matches(self.scene_classes, self.style_classes)
        else:
            self.matches = override_matches
        
        # TODO: fixed
        # if color_transfer:
        #     self.gaussian_masks = self.get_gaussian_masks(self.scene_weights)
        #     self.color_transfer(self.gaussian_masks)
        
        
        # self.original_style_image = self.style_image
        # self.original_style_masks = self.style_masks
        # if self.style_type == "single_prompt":
        #     # if erode or isolate:
        #     self.postprocess(erode, isolate)
        
        # render_RGBcolor_images("./debug/style.jpg", self.style_image)
        # render_depth_or_mask_images("./debug/style_mask.jpg", self.style_masks[0])
        # self.render_viewpoint(True, True, True, True)
        
        # exit()
        
    # def postprocess(self, erode, isolate):
        
        
    #     if isolate:
    #         final_style_masks = []
    #         final_style_image = []
    #     else:
    #         final_style_masks = torch.full_like(self.style_masks[0], -1, device=self.device)
    #         final_style_image = torch.zeros_like(self.style_image_list[0])
        
    #     for i in range(self.style_classes):
            
    #         original_mask = (self.style_masks[0] == i)
    #         erode_mask = original_mask.cpu().numpy().astype(np.uint8) * 255
    #         if erode:
    #             kernel_size = 5
    #             kernel = np.ones((kernel_size, kernel_size), np.uint8)
    #             erode_mask = cv2.erode(erode_mask, kernel, iterations=1)
    #         erode_mask = (erode_mask / 255).astype(bool)
            
            
    #         if isolate:
    #             cur_style_image = torch.full_like(self.style_image_list[0], 0, device=self.device)
    #             cur_style_image[:, original_mask] = self.style_image_list[0][:, original_mask]
    #             final_style_image.append(cur_style_image)
                
    #             cur_style_masks = torch.full_like(self.style_masks[0], -1, device=self.device)
    #             cur_style_masks[erode_mask] = i
    #             final_style_masks.append(cur_style_masks)
    #         else:
    #             final_style_image[:, original_mask] = self.style_image_list[0][:, original_mask]
    #             final_style_masks[erode_mask] = i
            
        
        
    #     # for i in range(self.style_classes):
            
            
    #     tmp_image = torch.zeros_like(self.style_image_list[0])
    #     if isolate:
    #         self.style_image_list = final_style_image
    #         self.style_masks = final_style_masks
            
    #         print(self.style_image_list[0].shape)
    #         print(torch.sum(self.style_masks[0] == -1))
    #         print(torch.sum(self.style_masks[0] == 0))
    #         print(torch.sum(self.style_masks[1] == 1))
    #         tmp_image[:, self.style_masks[0] == -1] = torch.tensor([[0], [0], [0]], device="cuda") / 255.0
    #         tmp_image[:, self.style_masks[0] == 0] = torch.tensor([[120], [183], [201]], device="cuda") / 255.0
    #         tmp_image[:, self.style_masks[1] == 1] = torch.tensor([[229], [139], [25]], device="cuda") / 255.0
    #     else:
    #         self.style_image_list = [final_style_image]
    #         self.style_masks = [final_style_masks]
        
        # print(self.style_masks[0, 294], self.style_masks[0, 295])
        # for i in range(-1, self.style_classes):
        #     print(torch.sum(self.style_masks == i))
        # print(f"inpaint{erode}{isolate}.jpg")
        # print(f"mask{erode}{isolate}.jpg")
        # render_RGBcolor_images(f"inpaint{erode}{isolate}.jpg", self.style_image)
        # render_RGBcolor_images(f"mask{erode}{isolate}.jpg", tmp_image)
        # exit()
        # print(self.style_masks.shape)
        
    # 生成depth的地方    
    def process_depth_images(self, gaussians):
        """
        Renders depth images and saves them into the viewpoint stack.
        
        @param gaussians: Gaussian object for rendering.
        """
        for _, view in enumerate(self.viewpoint_stack):
            depth_image = render(view, gaussians, self.pipe, self.bg)["depth"]
            view.depth_image = depth_image.squeeze().detach()
            

    def process_style_images(self, style_images_path, style_prompt):
        """
        Reads style images and concatenates them if needed.
        
        @param style_images_path: List of paths to the style images.
        @param style_prompt: String of the style prompt.
        """
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff'}
        self.style_image_list = []
        # TODO: change path to dir_path
        for root, dirs, files in os.walk(style_images_path):
            for file in files:
                if os.path.splitext(file)[1].lower() in image_extensions:
                    self.style_image_list.append(self.read_and_resize_image(os.path.join(root, file), 256))
        # for _, p in enumerate(style_images_path):
        #     self.style_image_list.append(self.read_and_resize_image(p, 256))
        # self.style_image = torch.cat(self.style_image_list, dim=2)
        
        # if len(style_images_path) == 1 and style_prompt is None:
        self.style_type = "single"
        # elif len(style_images_path) == 1 and style_prompt is not None:
        #     self.style_type = "single_prompt"
        # else:
        #     self.style_type = "multiple"

    def get_scene_weights(self, scene_prompt):
        """
        Segments scene images for weights or loads existing weights.
        If no scene prompt, return None
        Check if the file ./util_data/scene/{scene_name}/{scene_prompt}.pt exists.
        
        @param scene_prompt: String of the scene prompt.
        @return: Tensor of scene weights. Shape: [N, scene_classes, H, W]
        """
        
        if scene_prompt is None:
            return None
        
        scene_labels = split_prompt(scene_prompt)
        
        scene_name = os.path.basename(self.scene.model_path)
        goal_path = f"./util_data/scene/{scene_name}/{scene_prompt}.pt"
        
        if os.path.exists(goal_path):
            return torch.load(goal_path).cuda()
        
        os.makedirs(os.path.dirname(goal_path), exist_ok=True)
        
        if self.model is None:
            self.model = LangSAM()

        total_weights = []
        
        for cur in tqdm(range(len(self.viewpoint_stack))):
            viewpoint_cam = self.viewpoint_stack[cur]
            # cur_masks[i, j] == 0 denotes (i, j) is classified to "others"
            pil_image = ToPILImage(mode="RGB")(viewpoint_cam.original_image)
            
            cur_weights = []
            
            for i, label in enumerate(scene_labels):
                masks, _, _, _ = self.model.predict(pil_image, label)
                if len(masks) == 0:
                    print(f"There's no {label} in this image")
                    continue
                masks, _ = torch.max(masks, dim=0)
                cur_weights.append(masks)
                
            # cur_weights: [scene_classes, Hc, Wc]
            cur_weights = torch.stack(cur_weights)
            total_weights.append(cur_weights)
            
        total_weights = torch.stack(total_weights)
        torch.save(total_weights, goal_path)
        
        return total_weights


    def get_scene_masks(self, scene_weight):
        """
        Gets the binary mask from the scene weights through a threshold.
        If no scene prompt, all the pixels are "others".
        
        @param scene_weight: Tensor of scene weights. Shape: [N, H, W]
        @return: Tensor of scene masks. Shape: [N, H, W]
        """
        n = len(self.viewpoint_stack)
        _, h, w = self.viewpoint_stack[0].original_image.shape
        
        if scene_weight is None:
            return torch.zeros((n, h, w))
        
        total_masks = []
        for i, cur_image_weight in enumerate(scene_weight):
            cur_mask = torch.zeros((h, w))
            for j, cur_class_weight in enumerate(cur_image_weight):
                cur_mask[cur_class_weight > 0.7] = j + 1
            total_masks.append(cur_mask)
            
        total_masks = torch.stack(total_masks)
        
        return total_masks

        

    def get_gaussian_masks(self, scene_weights):
        """
        Unprojects the pixels to generate Gaussian masks for the scene images.
        If no scene prompt, all the pixels are "others".
        
        @param scene_weights: Tensor of scene weights. Shape: [N, H, W]
        @return: Tensor of Gaussian masks. Shape: [N]
        """
        
        if scene_weights is None:
            return torch.zeros_like(self.scene.gaussians._opacity, device=self.device)
    
        scene_weights = scene_weights.to(dtype=torch.float32, device="cuda")
        scene_weights = scene_weights.unsqueeze(2)
        
        # ignore 0: others
        gs_mask = torch.zeros_like(self.scene.gaussians._opacity)
        for cur in range(self.scene_classes - 1):

            weights = torch.zeros_like(self.scene.gaussians._opacity)
            weights_cnt = torch.zeros_like(self.scene.gaussians._opacity, dtype=torch.int32)

            for i, viewpoint_cam in enumerate(self.viewpoint_stack):
                self.scene.gaussians.apply_weights(viewpoint_cam, weights, weights_cnt, scene_weights[i][cur])
                
            weights /= weights_cnt + 1e-7
            gs_mask[weights > 0.7] = cur + 1
            
        return gs_mask.squeeze()


    def get_style_masks(self, style_prompt):
        """
        Gets the mask for the style images. If no style prompt, all the pixels are "others".
        
        @param style_prompt: String of the style prompt.
        @return: Tensor of style masks. Shape: [H, W]
        """
        
        if self.style_type == "single":
            _, h, w = self.style_image_list[0].shape
            return [torch.zeros((h, w), device=self.device)]
    
        # # if self.style_type == "multiple":
        # #     style_masks = []
        # #     for i, image in enumerate(self.style_image_list):
        # #         style_masks.append(torch.full((image.shape[-2:]), i))
        # #     return style_masks
        
        # # assert self.style_type == "single_prompt"
        
        # style_name = os.path.basename(self.scene.style_path)
        # style_labels = split_prompt(style_prompt)
        # goal_path = f"./util_data/style/{style_name}/{style_prompt}.pt"
        
        # if os.path.exists(goal_path):
        #     return [torch.load(goal_path).cuda()]
        
        # os.makedirs(os.path.dirname(goal_path), exist_ok=True)
        
        # if self.model is None:
        #     self.model = LangSAM()
            
        # _, h, w = self.style_image_list[0].shape
        # pil_image = ToPILImage(mode="RGB")(self.style_image_list[0])
        # style_mask = torch.zeros((h, w))
        
        
        # for i, label in enumerate(style_labels):
        #     masks, _, _, _ = self.model.predict(pil_image, label)
        #     if len(masks) == 0:
        #         print(f"There's no {label} in this image")
        #         continue
        #     masks, _ = torch.max(masks, dim=0)
        #     style_mask[masks > 0.7] = i + 1 

        # torch.save(style_mask, goal_path)
        # return [style_mask]

    def color_transfer(self, gaussian_masks):
        """
        Calculates mean color of each class in style image and assigns it to corresponding scene class.

        @param gs_masks: Tensor of Gaussian masks. Shape: [N, H, W]
        """
        origin_images = []
        for i, view in enumerate(self.viewpoint_stack):
            image = view.original_image.permute(1, 2, 0)
            origin_images.append(image)
            
        origin_images = torch.stack(origin_images).to(device=self.device)
        
        for i in range(self.scene_classes):            
                
            style_pixels = []
            # TODO:
            # for j, style_image in enumerate(self.style_image_list):
            #     style_pixels.append(style_image[:, :, :].permute(1, 2, 0))
            style_pixels.append(self.style_image_list[0][:, :, :].permute(1, 2, 0))
            style_pixels = torch.cat(style_pixels, dim=0)
            
            image_pixels = origin_images[self.scene_masks == i, :]
            
            image_set, color_tf = self.match_colors(image_pixels, style_pixels)
            origin_images[self.scene_masks == i, :] = image_set
            
            # self.scene.gaussians.apply_ct(color_tf.detach().cpu().numpy(), gaussian_masks == i)
            
        
        origin_images = origin_images.permute(0, 3, 1, 2)
        
        for i, cam in enumerate(self.viewpoint_stack):
            cam.original_image = origin_images[i]
            
        
        
    def match_colors(self, scene_images, style_image):
        """
        Transfers the style images' color to the scene images.
        
        @param scene_images: Tensor of scene images. Shape: [N, 3].
        @param style_image: Tensor of style images. Shape: [N, 3].
        @return: Transferred scene images and color transfer matrix.
        """
        sh = scene_images.shape
        image_set = scene_images.view(-1, 3)
        style_img = style_image.view(-1, 3).to(image_set.device)

        mu_c = image_set.mean(0, keepdim=True)
        mu_s = style_img.mean(0, keepdim=True)

        cov_c = torch.matmul((image_set - mu_c).transpose(1, 0), image_set - mu_c).float() / float(image_set.size(0))
        cov_s = torch.matmul((style_img - mu_s).transpose(1, 0), style_img - mu_s).float() / float(style_img.size(0))

        u_c, sig_c, _ = torch.svd(cov_c)
        u_s, sig_s, _ = torch.svd(cov_s)

        u_c_i = u_c.transpose(1, 0)
        u_s_i = u_s.transpose(1, 0)

        scl_c = torch.diag(1.0 / torch.sqrt(torch.clamp(sig_c, 1e-8, 1e8)))
        scl_s = torch.diag(torch.sqrt(torch.clamp(sig_s, 1e-8, 1e8)))

        tmp_mat = u_s @ scl_s @ u_s_i @ u_c @ scl_c @ u_c_i
        tmp_vec = mu_s.view(1, 3) - mu_c.view(1, 3) @ tmp_mat.T

        image_set = image_set @ tmp_mat.T + tmp_vec.view(1, 3)
        image_set = image_set.contiguous().clamp_(0.0, 1.0).view(sh)

        color_tf = torch.eye(4).float().to(tmp_mat.device)
        color_tf[:3, :3] = tmp_mat
        color_tf[:3, 3:4] = tmp_vec.T
        return image_set, color_tf
    
    
    def unify_scene_images_size(self):
        """
        Resizes all scene images to be the same size.
        """
        min_h, min_w = 10000, 10000
        for i, view in enumerate(self.viewpoint_stack):
            min_h = min(min_h, view.image_height)
            min_w = min(min_w, view.image_width)
            
        for i, view in enumerate(self.viewpoint_stack):
            view.original_image = view.original_image[:, :min_h, :min_w]
            view.image_width, view.image_height = min_w, min_h
            
    def read_and_resize_image(self, image_path, target_height=256):
        """
        Reads and resizes an image from a file path.
        
        @param image_path: Path to the image file.
        @param size: Desired size for resizing.
        @return: Tensor of the resized image. Shape: [C, H, W]
        """
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        original_height, original_width = image_rgb.shape[:2]
        scale_ratio = target_height / original_height
        target_width = int(original_width * scale_ratio)
        
        resized_image = cv2.resize(image_rgb, (target_width, target_height), interpolation=cv2.INTER_AREA)
        resized_image = torch.from_numpy(resized_image).permute(2, 0, 1).float().to(device=self.device) / 255.0
        return resized_image

    def render_viewpoint(self, if_depth, if_mask, if_original, if_render, path="./debug"):
        """
        Renders viewpoints and saves corresponding images.
        
        @param render_scene: Boolean to render scene images.
        @param render_gaussian: Boolean to render Gaussian images.
        @param render_depth: Boolean to render depth images.
        @param render_final: Boolean to render final composite images.
        """
        
        depth_path = os.path.join(path, "depth/")
        mask_path = os.path.join(path, "mask/")
        original_path = os.path.join(path, "original/")
        render_path = os.path.join(path, "render/")
        
        os.makedirs(depth_path, exist_ok=True)
        os.makedirs(mask_path, exist_ok=True)
        os.makedirs(original_path, exist_ok=True)
        os.makedirs(render_path, exist_ok=True)
        
        for i, view in enumerate(self.viewpoint_stack):
            images_pkgs = render(view, self.scene.gaussians, self.pipe, self.bg)
            
            if if_depth:
                depth_image = images_pkgs["depth"]
                cur_depth_path = os.path.join(depth_path, f"{int(i):04d}.png")
                render_depth_or_mask_images(cur_depth_path, depth_image)
            
            if if_mask:
                mask_image = self.scene_masks[i]
                cur_mask_path = os.path.join(mask_path, f"{int(i):04d}.png")
                render_depth_or_mask_images(cur_mask_path, mask_image)
                
            if if_original:
                original_image = view.original_image
                cur_original_path = os.path.join(original_path, f"{int(i):04d}.png")
                render_RGBcolor_images(cur_original_path, original_image)
                
            if if_render:
                render_image = images_pkgs["render"]
                cur_render_path = os.path.join(render_path, f"{int(i):04d}.png")
                render_RGBcolor_images(cur_render_path, render_image)
            
            
            