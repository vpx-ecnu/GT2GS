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
    network_gui.init(config.app.ip, config.app.port)
    torch.autograd.set_detect_anomaly(config.app.detect_anomaly)
    
    trainer = StyleTrainer(config)
    trainer.train()
    
    # render_viewpoint(trainer)


if __name__ == "__main__":
    main()