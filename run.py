from dataclasses import dataclass
from omegaconf import OmegaConf
from pathlib import Path
import subprocess
from icecream import ic
import os
import shutil
import torch

@dataclass
class RunConfig:
    scene_type: str = "llff"
    scene_name: str = "fern"
    style_type: str = "style"
    style_name: str = "14.jpg"
    override: bool = False
    
@dataclass
class RunPath:
    source_path: Path
    model_path: Path
    style_img: Path
    start_checkpoint: Path
    resolution: str = "4"
    
    def __init__(self, config: RunConfig):
        
        # set source path    
        source_path_dict = {
            "llff": Path("./data/original_data/llff") / config.scene_name,
            "tnt": Path("./data/preprocessed_data/tnt") / config.scene_name
        }
        self.source_path = source_path_dict[config.scene_type]
        
        # set style image path
        style_img_dict = {
            "style": Path("./data/original_data/styles") / config.style_name,
            "texture": Path("./data/original_data/new_tex") / config.style_name
        }
        self.style_img = style_img_dict[config.style_type]
        
        base_model_path_dict = {
            "llff": Path("/Datasets/") / "ckpt" / "radegs_0" / "original_llff" / config.scene_name,
            "tnt": Path("/Datasets/") / "ckpt" / "radegs_0" / "tnt" / config.scene_name
        }
        self.base_model_path = base_model_path_dict[config.scene_type]
        
        config_path_dict = {
            "llff": Path("./configs/") / "llff_single.yaml",
            "tnt": Path("./configs/") / "tnt_single.yaml"
        }
        self.config_path = config_path_dict[config.scene_type]
        
        
        resolution_dict = { "llff": "4", "tnt": "1"}
        self.resolution = resolution_dict[config.scene_type]
        self.stylized_model_path = Path("./output/") / config.scene_type / config.scene_name / config.style_type / config.style_name
        
        self.result_path = Path("/Datasets/AAAI26/GT2GS/GT2GS") \
            / config.scene_type / config.scene_name / config.style_type / config.style_name
        self.source_train_frames_path = self.stylized_model_path / "render"
        self.source_video_path = self.stylized_model_path / "video" / "video.mp4"
        self.target_train_frames_path = self.result_path / "train_frames"
        self.target_video_path = self.result_path / "video.mp4"
        
    
def stylize(config: RunConfig, run_path: RunPath):
    
    if os.path.exists(run_path.target_video_path):
        if not config.override:
            return
        
    ic("Stylize")
    
    subprocess.run([
        "python", "style_transfer.py",
        "--config", str(run_path.config_path),
        "--source_path", str(run_path.source_path),
        "--model_path", str(run_path.base_model_path),
        "--style_images", str(run_path.style_img),
        "--resolution", run_path.resolution,
        "--stylized_model_path", str(run_path.stylized_model_path),
    ])
    
def render(config: RunConfig, run_path: RunPath):
    
    if os.path.exists(run_path.target_video_path):
        if not config.override:
            return
        
    ic("Render Train Views")
    
    config_path = run_path.stylized_model_path / "config.yaml"
    
    subprocess.run([
        "python", "scripts/render_scene.py",
        "--config", config_path,
        "--resolution", run_path.resolution,
    ])
    
    ic("Render Video")
    
    subprocess.run([
        "python", f"scripts/render_{config.scene_type}_video.py",
        "--config", config_path,
        "--resolution", run_path.resolution,
    ])
    
    
def get_results(config: RunConfig, run_path: RunPath):
    
    if os.path.exists(run_path.target_video_path):
        if not config.override:
            return
    
    os.makedirs(run_path.target_train_frames_path, exist_ok=True)
    train_frames = sorted(os.listdir(run_path.source_train_frames_path))
    
    shutil.move(run_path.source_video_path, run_path.target_video_path)
    for file_name in train_frames:
        shutil.move(run_path.source_train_frames_path / file_name,
                    run_path.target_train_frames_path / file_name)
    
    shutil.rmtree(run_path.source_train_frames_path.parent)
    
def main():
    config = OmegaConf.structured(RunConfig)
    config = OmegaConf.merge(config, OmegaConf.from_cli())
    run_path = RunPath(config)
    
    stylize(config, run_path)
    render(config, run_path)
    get_results(config, run_path)
    
    
if __name__ == "__main__":
    main()