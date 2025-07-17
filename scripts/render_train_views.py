import sys
sys.path.append("./gs")
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
workspace_dir = os.path.dirname(script_dir)
sys.path.append(workspace_dir)

from gt2gs.style_utils import render_RGBcolor_images
from gt2gs.style_config import parse_args
from gt2gs.style_trainer import StyleTrainer
from tqdm import tqdm

    
if __name__ == '__main__':
    
    config = parse_args()
    config.model.model_path = config.style.stylized_model_path
    
    trainer = StyleTrainer(config)
    
    path = os.path.join(trainer.config.model.model_path, "train")
    os.makedirs(path, exist_ok=True)
    
    for i, view in enumerate(tqdm(trainer.scene.getTrainCameras(), desc="Rendering progress")):
        img = trainer.get_render_pkgs(view)["render"]
        cur_render_path = os.path.join(path, f"{int(i):04d}.png")
        render_RGBcolor_images(cur_render_path, img)