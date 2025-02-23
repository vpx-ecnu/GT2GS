# style_trainer.py
from typing import Dict, List
import torch
from scene import GaussianModel, Scene
from style_config import ConfigManager
from style_utils import CUDATimer

from gaussian_renderer import network_gui, render
from style_observer import TrainingMetrics
from style_phase import TrainingPhaseType, TrainingPhase, StylizationPhase, ColorTransferPhase
from style_observer import TrainingObserver, ProgressTracker, CheckpointSaver
from style_preprocess import StyleContext



class StyleTrainer:
    
    def __init__(self, config: ConfigManager):
        self.config = config
        self.device = config.model.data_device
        self.timer = CUDATimer()
        self.cur_phase = None
        
        self._init_components()
        self._init_phases()
        
    def train(self):
        
        self._init_observers()
        
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
        
        if self.config.style.color_transfer:
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
    
            
    def get_render_pkgs(self, viewpoint_cam):
        return render(viewpoint_cam, self.gaussians, self.config.pipe, self._get_background())
    
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

