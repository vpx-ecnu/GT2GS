# style_main.py
import torch
from style_config import parse_args
from utils.general_utils import safe_state
from gaussian_renderer import network_gui
from style_trainer import StyleTrainer
import random
import numpy as np
def main():
    
    seed = 42
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    config = parse_args()   
     
    safe_state(config.app.quiet)
    network_gui.init(config.app.ip, config.app.port)
    torch.autograd.set_detect_anomaly(config.app.detect_anomaly)
    
    trainer = StyleTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()