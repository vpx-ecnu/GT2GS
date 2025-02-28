# style_config.py
from dataclasses import dataclass
from simple_parsing import ArgumentParser, field
from simple_parsing.helpers import list_field
import os

@dataclass
class ModelConfig:
    sh_degree: int = 3
    source_path: str = field(None, alias="-s")
    original_model_path: str = field(None, alias="-o")  # If this field is specified, then load the model for this path
    model_path: str = field(None, alias="-m") # If render only, please specify this field
    images: str = field("images", alias="-i")
    resolution: int = field(-1, alias="-r")
    white_background: bool = field(False, action="store_true")
    data_device: str = "cuda"
    eval: bool = field(False, action="store_true")
    
@dataclass
class PipelineConfig:
    convert_SHs_python: bool = field(False, action="store_true")
    compute_cov3D_python: bool = field(False, action="store_true")
    debug: bool = field(False, action="store_true")
    
@dataclass
class OptimizationConfig:
    
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
    random_background: bool = field(False, action="store_true")
    

@dataclass
class ApplicationConfig:
    
    ip: str = "127.0.0.1"
    port: int = 6009
    debug_from: int = -1
    detect_anomaly: bool = field(False, action="store_true")
    quiet: bool = field(False, action="store_true")
    
@dataclass
class CheckpointConfig:
    save_iterations: list[int] = list_field()
    checkpoint_iterations: list[int] = list_field()
    start_checkpoint: str = None

@dataclass
class StyleConfig:
    
    name: str = "deafult"
    prior: bool = field(False, action="store_true")
    lambda_consistent_loss: float = 2
    lambda_prior_loss: float = 2
    lambda_content_loss: float = 0.005
    lambda_imgtv_loss: float = 0.02
    lambda_depth_loss: float = 0.05
    
    color_transfer: bool = field(False, action="store_true")
    color_transfer_iter: int = 400
    style_iter: int = 600
    
    style_image: str = None
    style_image_size: int = 256
    
class ConfigManager:
    
    model: ModelConfig
    opt: OptimizationConfig
    pipe: PipelineConfig
    app: ApplicationConfig
    style: StyleConfig
    ckpt: CheckpointConfig
    
    def __init__(self, raw_config):
        self.model = raw_config.model
        self.opt = raw_config.opt
        self.pipe = raw_config.pipe
        self.app = raw_config.app
        self.style = raw_config.style
        self.ckpt = raw_config.ckpt
        
        self._generate_output_path()
        self._process_iteration()
        self._save_args()
        
    def set_debug(self, val):
        self.pipe.debug = val
    
    def _generate_output_path(self):
        if self.model.model_path is not None:
            return
        
        scene = os.path.basename(os.path.basename(self.model.source_path.rstrip("/")))
        style = os.path.splitext(os.path.basename(self.style.style_image))[0]
        self.model.model_path = f"output/style/{scene}/{style}/"
    
    # TODO: 需要能够交替的优雅实现
    def _process_iteration(self):
        
        self.opt.iterations = (self.style.color_transfer_iter * 2 if self.style.color_transfer else 0)
        self.opt.iterations += self.style.style_iter 
        
        self.opt.densify_from_iter = 1
        self.opt.densify_until_iter = self.style.color_transfer_iter if self.style.color_transfer else 1
        
        self.opt.style_from_iter = 1 + (self.style.color_transfer_iter if self.style.color_transfer else 0)
        self.opt.style_until_iter = self.opt.style_from_iter + self.style.style_iter - 1

    def _save_args(self):
        
        from argparse import Namespace  # for compatibility
        print(f"Training on {self.model.source_path}")
        print(f"Original Gaussian model path: {self.model.original_model_path}")
        print(f"Stylized Gaussian model output path: {self.model.model_path}")
        
        os.makedirs(self.model.model_path, exist_ok=True)
        with open(os.path.join(self.model.model_path, "cfg_args"), "w") as cfg_log_f:
            cfg_log_f.write(str(Namespace(**vars(self.model))))
    
    
def parse_args():
    parser = ArgumentParser(description="Training script parameters")
    parser.add_arguments(ModelConfig, dest="model")
    parser.add_arguments(OptimizationConfig, dest="opt")
    parser.add_arguments(PipelineConfig, dest="pipe")
    parser.add_arguments(ApplicationConfig, dest="app")
    parser.add_arguments(CheckpointConfig, dest="ckpt")
    parser.add_arguments(StyleConfig, dest="style")
    
    config = parser.parse_args()
    
    return ConfigManager(config)
    