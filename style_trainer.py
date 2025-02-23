from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum, auto
from typing import Dict, List, Optional, Deque
from collections import deque
from tqdm import tqdm
import torch
from scene import GaussianModel, Scene
from style_config import ConfigManager
from style_utils import CUDATimer

from gaussian_renderer import network_gui, render
import os
from random import randint
import time

class TrainingPhaseType(Enum):
    COLOR_TRANSFER_1 = auto()
    STYLIZATION = auto()
    COLOR_TRANSFER_2 = auto()
    

class TrainingPhase(ABC):
    def __init__(self, trainer, start_iter, end_iter):
        self.trainer = trainer
        self.config = trainer.config
        self.start_iter = start_iter
        self.end_iter = end_iter
        self.viewpoint_stack = None

    def on_phase_start(self): ...

    @abstractmethod
    def on_iteration(self, iteration: int) -> Dict[str, torch.Tensor]: ...

    def on_phase_end(self): ...
    
    @abstractmethod
    def _get_viewpoint_cam(self): ...
    
    @abstractmethod
    def _densification(self, iteration: int): ...


class ColorTransferPhase(TrainingPhase):
    
    @torch.no_grad
    def on_phase_start(self):
        
        if not self.config.style.color_transfer:
            return 


    def on_iteration(self, iteration: int) -> Dict[str, torch.Tensor]:
        
        viewpoint_cam = self._get_viewpoint_cam()
        
        with self.trainer.timer as timer:
            
            time.sleep(0.005)
            # self._densify_gaussians(iteration)

        return {"no": 1}, timer.elapsed_ms
    
    def _get_viewpoint_cam(self):
        if not self.viewpoint_stack:
            self.viewpoint_stack = self.trainer.scene.getTrainCameras().copy()
        return self.viewpoint_stack.pop(randint(0, len(self.viewpoint_stack) - 1))

    def _densification(self, iteration: int):
        return

class StylizationPhase(TrainingPhase):
    
    def on_iteration(self, iteration: int) -> Dict[str, torch.Tensor]:
        
        viewpoint_cam = self._get_viewpoint_cam()
        
        with self.trainer.timer as timer:
            
            time.sleep(0.005)
            

        return {"no": 1}, timer.elapsed_ms

    def _get_viewpoint_cam(self):
        if not self.viewpoint_stack:
            self.viewpoint_stack = self.trainer.scene.getTrainCameras()
            self.viewpoint_len = len(self.viewpoint_stack)
            self.viewpoint_idx = -1
        self.viewpoint_idx = (self.viewpoint_idx + 1) % self.viewpoint_len
        return self.viewpoint_stack[self.viewpoint_idx]
    
    
    def _densification(self, iteration: int):
        return
    

@dataclass
class TrainingMetrics:
    iteration: int
    phase: TrainingPhaseType
    losses: Dict[str, float]
    timing: float
    
    
class TrainingObserver(ABC):
    
    def on_iteration_start(self, iteration: int): ...
    
    def on_iteration_end(self, metrics: TrainingMetrics): ...
    
    def on_phase_changed(self, previous: TrainingPhaseType, current: TrainingPhaseType): ...
    
    def on_training_end(self): ...
    

class ProgressTracker(TrainingObserver):
    
    def __init__(self, trainer: 'StyleTrainer'):
        
        self.bar_format = "{l_bar}{bar:50}{r_bar}"
        self.trainer = trainer
        self.global_pbar = tqdm(
            total=trainer.config.opt.iterations, 
            desc=f"{'Global Process':<16}",
            bar_format=self.bar_format
        )
        self.phase_bars: Dict[TrainingPhaseType, tqdm] = {}
        self.current_phase: Optional[TrainingPhaseType] = None
        self.update_interval: int = 10

    def on_phase_changed(self, previous: TrainingPhaseType, current: TrainingPhaseType):
        if previous in self.phase_bars:
            self.phase_bars[previous].close()
        
        if current not in self.phase_bars:
            phase = self.trainer.phases[current]
            self.phase_bars[current] = tqdm(
                total=phase.end_iter - phase.start_iter + 1,
                desc=f"{current.name.replace('_', ' ').title():<16}",
                bar_format=self.bar_format
            )

    def on_iteration_end(self, metrics: TrainingMetrics):
        
        if metrics.iteration % self.update_interval != 0:
            return 
        
        self.global_pbar.update(self.update_interval)
        self.global_pbar.set_postfix(metrics.losses)
        
        if metrics.phase in self.phase_bars:
            self.phase_bars[metrics.phase].update(self.update_interval)
            self.phase_bars[metrics.phase].set_postfix(metrics.losses)
            
    def on_training_end(self):
        
        self.global_pbar.close()
        for phase_bar in self.phase_bars.values():
            phase_bar.close()
            
            
class CheckpointSaver(TrainingObserver):
    
    def __init__(self, trainer: 'StyleTrainer'):
        self.trainer = trainer

    def on_iteration_end(self, metrics: TrainingMetrics):
        
        if metrics.iteration in self.trainer.config.ckpt.checkpoint_iterations:                
            print("\n[ITER {}] Saving Checkpoint".format(metrics.iteration))
            torch.save(
                (self.trainer.gaussians.capture(), metrics.iteration),
                self.trainer.config.model.model_path + "/chkpnt" + str(metrics.iteration) + ".pth",
            )
            
    def on_training_end(self):
        print("\n[ITER {}] Saving Gaussians".format(self.trainer.config.opt.iterations))
        self.trainer.scene.save(self.trainer.config.opt.iterations, self.trainer.config.model.model_path)
        


class StyleTrainer:
    
    def __init__(self, config: ConfigManager):
        self.config = config
        self.device = config.model.data_device
        self.timer = CUDATimer()
        self.cur_phase = None
        
        self._init_components()
        self._init_phases()
        self._init_observers()
        
    def train(self):
        
        for self.iteration in range(1, self.config.opt.iterations + 1):
            self._handle_gui()
            self._train_iteration()
            
        
        for observer in self.observers:
            observer.on_training_end()
    
        
    def _train_iteration(self):
        
        new_phase = self._get_train_phase()
        if new_phase != self.cur_phase:
            self._switch_phase(new_phase)
        phase = self.phases[self.cur_phase]
        
        for observer in self.observers:
            observer.on_iteration_start(self.iteration)
        
        losses, timing = phase.on_iteration(self.iteration)
        
        metrics = TrainingMetrics(
            iteration=self.iteration,
            phase=self.cur_phase,
            losses=losses,
            timing=timing
        )
        
        for observer in self.observers:
            observer.on_iteration_end(metrics)
        
        
    
    def _switch_phase(self, new_phase):
        
        if self.cur_phase is not None:
            self.phases[self.cur_phase].on_phase_end()
        
        self.cur_phase = new_phase
        self.phases[self.cur_phase].on_phase_start()
        
        for observer in self.observers:
            observer.on_phase_changed(self.cur_phase, new_phase)

        
    def _init_components(self):
        self.gaussians = GaussianModel(self.config.model.sh_degree)
        self.scene = Scene(self.config.model, self.gaussians, -1, shuffle=False)
        
        bg_color = [1, 1, 1] if self.config.model.white_background else [0, 0, 0]
        self.background = torch.tensor(bg_color, dtype=torch.float32, device=self.config.model.data_device)
        
        self.first_iter = 1
        if self.config.ckpt.start_checkpoint is not None:
            self._load_checkpoint()
            
        self.gaussians.training_setup(self.config.opt)
        
        
    def _init_phases(self):
        
        self.phases: Dict[TrainingPhaseType, TrainingPhase] = {
            TrainingPhaseType.STYLIZATION: 
                StylizationPhase(self, self.config.opt.style_from_iter, 
                                       self.config.opt.style_until_iter),
        }
        
        if not self.config.style.color_transfer:
            return 
        
        self.phases.update({
            TrainingPhaseType.COLOR_TRANSFER_1: 
                ColorTransferPhase(self, 1, 
                                         self.config.opt.style_from_iter - 1),
            TrainingPhaseType.COLOR_TRANSFER_2: 
                ColorTransferPhase(self, self.config.opt.style_until_iter + 1, 
                                         self.config.opt.iterations)  
        })
        
        

    def _init_observers(self):
        
        self.observers: List[TrainingObserver] = [
            ProgressTracker(self),
            CheckpointSaver(self)
        ]
        
        
    def _get_train_phase(self) -> TrainingPhaseType:
        
        for phase_type, phase in self.phases.items():
            if phase.start_iter <= self.iteration <= phase.end_iter:
                return phase_type
        raise ValueError(f"Iteration {self.iteration} out of phase range")
        
    def _load_checkpoint(self):
        (model_params, self.first_iter) = torch.load(self.config.ckpt.start_checkpoint)
        self.first_iter += 1
        self.gaussians.restore(model_params, self.config.opt)
        

    def _get_background(self):
        if self.config.opt.random_background:
            return torch.rand((3), device=self.config.model.data_device)
        return self.background
    
    # original network gui handler
    def _handle_gui(self):
        if network_gui.conn is None:
            network_gui.try_connect()
        while network_gui.conn is not None:
            try:
                net_image_bytes = None
                (
                    custom_cam,
                    do_training,
                    self.config.pipe.convert_SHs_python,
                    self.config.pipe.compute_cov3D_python,
                    keep_alive,
                    scaling_modifer,
                ) = network_gui.receive()
                if custom_cam is not None:
                    net_image = render(
                        custom_cam, self.gaussians, self.config.pipe, self.background, scaling_modifer
                    )["render"]
                    net_image_bytes = memoryview(
                        (torch.clamp(net_image, min=0, max=1.0) * 255)
                        .byte()
                        .permute(1, 2, 0)
                        .contiguous()
                        .cpu()
                        .numpy()
                    )
                network_gui.send(net_image_bytes, self.config.model.source_path)
                if do_training and (
                    (self.iteration < int(self.config.opt.iterations)) or not keep_alive
                ):
                    break
            except Exception:
                network_gui.conn = None
