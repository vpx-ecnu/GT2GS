# style_main.py
import torch
from style_config import parse_args
from utils.general_utils import safe_state
from gaussian_renderer import network_gui
from style_trainer import StyleTrainer
import random
import numpy as np
import wandb 
from datetime import datetime
import gc

# 定义 WandB 超参数搜索配置
sweep_config = {
    "method": "bayes",  # 推荐使用贝叶斯优化
    "metric": {
        "name": "Loss",
        "goal": "minimize",
    },
    "parameters": {
        "position_lr_init": {"min": 1e-5, "max": 1e-3},
        "position_lr_final": {"min": 1e-7, "max": 1e-5},
        "position_lr_delay_mult": {"min": 0.001, "max": 0.1},
        "position_lr_max_steps": {"values": [20_000, 30_000, 40_000]},
        "feature_lr": {"min": 0.001, "max": 0.01},
        "opacity_lr": {"min": 0.01, "max": 0.1},
        "scaling_lr": {"min": 0.001, "max": 0.01},
        "rotation_lr": {"min": 0.0005, "max": 0.005}
    }
}

def train_sweep():
    # 初始化 WandB 并获取超参数
    wandb.init()
    
    # 设置随机种子（确保每次实验可复现）
    seed = 42
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    # 合并配置参数
    config = parse_args()
    
    # 用 WandB 的超参数覆盖原始配置
    config.position_lr_init = wandb.config.position_lr_init
    config.position_lr_final = wandb.config.position_lr_final
    config.position_lr_delay_mult = wandb.config.position_lr_delay_mult
    config.position_lr_max_steps = wandb.config.position_lr_max_steps
    config.feature_lr = wandb.config.feature_lr
    config.opacity_lr = wandb.config.opacity_lr
    config.scaling_lr = wandb.config.scaling_lr
    config.rotation_lr = wandb.config.rotation_lr

    try:
        # 初始化训练环境
        safe_state(config.app.quiet)
        torch.autograd.set_detect_anomaly(config.app.detect_anomaly)

        # 创建并运行训练器
        trainer = StyleTrainer(config)
        trainer.train()
        
    finally:
        if hasattr(trainer, "model"):
            del trainer.model
        if hasattr(trainer, "optimizer"):
            del trainer.optimizer
        del trainer
        
        # 清理 GPU 缓存
        torch.cuda.empty_cache()
        
        # 遍历删除残留张量
        for obj in gc.get_objects():
            if torch.is_tensor(obj) and obj.is_cuda:
                del obj
                
        gc.collect()
        torch.cuda.ipc_collect()
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            peak_mem_1 = torch.cuda.max_memory_allocated()
            print(peak_mem_1)
            torch.cuda.empty_cache()

def main():
    config = parse_args()
    
    sweep_id = wandb.sweep(sweep_config, project="Texture-GS")
    wandb.agent(sweep_id, function=train_sweep, count=100)


if __name__ == "__main__":
    main()