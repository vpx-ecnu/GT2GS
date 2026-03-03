# style_config.py
from dataclasses import dataclass
from simple_parsing import ArgumentParser, field, Serializable
from simple_parsing.helpers import list_field
from datetime import datetime
from zoneinfo import ZoneInfo
import os

@dataclass
class ModelConfig(Serializable):
    sh_degree: int = 3
    source_path: str = field(None, alias="-s")
    model_path: str = field(None, alias="-m")
    images: str = field("images", alias="-i")
    resolution: int = field(-1, alias="-r")
    white_background: bool = False
    data_device: str = "cuda"
    eval: bool = False

    view_num: int = -1
    drop_rate: float = 0
    
@dataclass
class PipelineConfig(Serializable):
    convert_SHs_python: bool = False
    compute_cov3D_python: bool = False
    debug: bool = False
    
@dataclass
class OptimizationConfig(Serializable):
    
    position_lr_init: float = 0.00016
    position_lr_final: float = 0.0000016
    position_lr_delay_mult: float = 0.01
    position_lr_max_steps: int = 30_000
    feature_lr: float = 0.0025
    opacity_lr: float = 0.05
    scaling_lr: float = 0.005
    rotation_lr: float = 0.001
    percent_dense: float = 0.01
    lambda_dssim: float = 0.2
    densification_interval: int = 50
    opacity_reset_interval: int = 10000  # not used
    
    densify_grad_threshold: float = 0.0002
    random_background: bool = False
    
    
    style_densification_interval: int = 50
    style_densification_threshold: float = 0.00001

@dataclass
class ApplicationConfig(Serializable):
    
    ip: str = "127.0.0.1"
    port: int = 6009
    debug_from: int = -1
    detect_anomaly: bool = False
    quiet: bool = False
    need_log: bool = False
    
    
@dataclass
class VideoConfig(Serializable):
    # For LLFF spiral Video
    num_rotations: int = 2
    
    # For Rendering Video
    enable_save_frames: bool = False
    save_frames_path: str = "video"
    num_frames: int = 120

@dataclass
class CheckpointConfig(Serializable):
    
    load_iterations: int = -1
    save_iterations: list[int] = list_field()
    checkpoint_iterations: list[int] = list_field()
    start_checkpoint: str = None

@dataclass
class StyleConfig(Serializable):
    
    stylized_model_path: str = field(None, alias="-o")
    
    name: str = "default"
    gta_type: str = "default"
    no_grad: bool = False
    
    
    lambda_consistent_loss: float = 0
    lambda_prior_loss: float = 2
    lambda_nnfm_loss: float = 2
    lambda_content_loss: float = 0.005
    lambda_imgtv_loss: float = 0.02
    lambda_depth_loss: float = 0
    lambda_shape_loss: float = 0.1
    lambda_delta_opacity: float = 0
    lambda_delta_scaling: float = 0
    lambda_delta_position: float = 1
    lambda_adaptive: float = 0.8
    
    enable_stylize_densify: bool = False
    pre_densify: bool = False
    enable_geometry_correction: bool = False  
    enable_nnfm_correction: bool = False
    enable_color_transfer: bool = False
    enable_feature_enhancement: bool = False
    enable_prior: bool = False
    enable_weighted: bool = False
    
    preprocess_iter: int = 400
    postpreprocess_iter: int = 400
    correction_iter: int = 400
    
    rounds: int = 10
    style_iter: int = 80
    geometry_iter: int = 0
    
    style_image: str = None
    style_image_size: int = 256
    
    init_densification_image_intervals: int = 10
    init_densification_downsample: int = 2
    
    depth_group_num: int = 3
    downscale_limit_ratio: int = 2
    
    theta: int = 0

@dataclass
class ConfigManager(Serializable):
    
    model: ModelConfig
    opt: OptimizationConfig
    pipe: PipelineConfig
    app: ApplicationConfig
    style: StyleConfig
    ckpt: CheckpointConfig
    video: VideoConfig
    
    def __init__(self, raw_config):
        self.model = raw_config.model
        self.opt = raw_config.opt
        self.pipe = raw_config.pipe
        self.app = raw_config.app
        self.style = raw_config.style
        self.ckpt = raw_config.ckpt
        self.video = raw_config.video
        
        self._generate_output_path()
        self._check_params()
        # self._process_iteration()
        self._save_args()
        
    def set_debug(self, val):
        self.pipe.debug = val
    
    def _generate_output_path(self):
        if self.style.stylized_model_path is not None:
            return
        
        current_date = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y-%m-%d_%H-%M-%S")
        self.style.stylized_model_path = os.path.join("output", current_date)
    
    def _check_params(self):
        # assert os.path.exists(self.style.style_image), f"{self.style.style_image} does not exists."
        
        
        assert self.style.theta >= 0 and self.style.theta <= 359, "theta mush between 0 and 359"
        
        if self.style.enable_prior:
            assert self.style.enable_feature_enhancement, "prior loss requires enable_feature_enhancement"

    def _save_args(self):
        
        print(f"Training on {self.model.source_path}")
        print(f"Original Gaussian model path: {self.model.model_path}")
        print(f"Stylized Gaussian model output path: {self.style.stylized_model_path}")
        
        os.makedirs(self.style.stylized_model_path, exist_ok=True)
        
        from argparse import Namespace  # for compatibility
        model_vars = vars(self.model).copy()
        model_vars['model_path'] = self.style.stylized_model_path
        with open(os.path.join(self.style.stylized_model_path, "cfg_args"), "w") as cfg_log_f:
            cfg_log_f.write(str(Namespace(**model_vars)))
            
        self.save_yaml(os.path.join(self.style.stylized_model_path, "config.yaml"))
    
    
def parse_args():
    parser = ArgumentParser(description="Training script parameters",
                            add_config_path_arg="config")
    
    parser.add_arguments(ModelConfig, dest="model")
    parser.add_arguments(OptimizationConfig, dest="opt")
    parser.add_arguments(PipelineConfig, dest="pipe")
    parser.add_arguments(ApplicationConfig, dest="app")
    parser.add_arguments(CheckpointConfig, dest="ckpt")
    parser.add_arguments(StyleConfig, dest="style")
    parser.add_arguments(VideoConfig,dest="video" )
    
    config = parser.parse_args()
    
    return ConfigManager(config)
    