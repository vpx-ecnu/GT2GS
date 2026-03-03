# style_main.py
import sys
sys.path.append("./gs/")

import torch
from gt2gs.style_config import parse_args
from gs.utils.general_utils import safe_state
from gs.gaussian_renderer import network_gui
from gt2gs.style_trainer import StyleTrainer
import random
import numpy as np
import wandb
from datetime import datetime
from gt2gs.style_utils import render_viewpoint

def main():
    
    seed = 42
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    config = parse_args()
    
    if config.app.need_log:
        wandb.init(
            project="Texture-GS",
            name = config.style.name,
            config=config,
            group=datetime.now().strftime("%Y-%m-%d-%H")
        )
    safe_state(config.app.quiet)
    # network_gui.init(config.app.ip, config.app.port)
    torch.autograd.set_detect_anomaly(config.app.detect_anomaly)
    
    trainer = StyleTrainer(config)

    
    # ##################################################################
    # torch.cuda.reset_peak_memory_stats()
    # starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    # starter.record()
    # ##################################################################
    trainer.train()
    
    # ####################################################################
    # ender.record()
    # torch.cuda.synchronize()
    # elapsed_time_s = starter.elapsed_time(ender) / 1000 # Convert to seconds
    # peak_mem_gb = torch.cuda.max_memory_allocated() / 1024**3  # Convert to GB

    # print(f"峰值显存占用: {peak_mem_gb:.2f} GB")
    # print(f"GPU 运行时间: {elapsed_time_s:.2f} s")

    # from pathlib import Path
    # with open(Path(config.style.stylized_model_path) / "metrics.txt", "w") as f:
    #     f.write(f"运行时间 (s): {elapsed_time_s:.2f}\n")
    #     f.write(f"峰值显存占用 (GB): {peak_mem_gb:.2f}\n")
    # ####################################################################
    # trainer.config.model.model_path = trainer.config.style.stylized_model_path
    # if trainer.config.model.source_path.find("tnt") != -1:
    #     from scripts.render_tnt_video import render_video
    # else:
    #     from scripts.render_llff_video import render_video
    # from scripts.render_llff_video import render_video
    # render_video(trainer)
    
    # render_viewpoint(trainer)


if __name__ == "__main__":
    main()