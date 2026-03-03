# style_trainer.py
from typing import Dict, List
import torch
from gs.scene import GaussianModel, Scene
from gt2gs.style_config import ConfigManager
from gt2gs.style_utils import CUDATimer

from gs.gaussian_renderer import network_gui, render
from gt2gs.style_observer import TrainingMetrics
from gt2gs.style_observer import TrainingObserver, ProgressTracker, CheckpointSaver
from gt2gs.style_preprocess import preprocess

from gt2gs.phase.geometry_phase.preprocess_phase import PreProcessPhase
from gt2gs.phase.geometry_phase.postprocess_phase import PostProcessPhase
from gt2gs.phase.geometry_phase.correction_phase import CorrectionPhase
from gt2gs.phase.stylize_phase.nnfm_phase import NNFMPhase
from gt2gs.phase.stylize_phase.prior_phase import PriorPhase
from icecream import ic

from gt2gs.style_utils import *

class StyleTrainer:
    
    def __init__(self, config: ConfigManager):
        self.config = config
        self.device = config.model.data_device
        self.meta_information = []
        self._init_components()
        
    def train(self):
        
        preprocess(self)
        self.timer = CUDATimer()
        self.cur_phase = -1
        self._init_phases()
        self._init_observers()
        
        for self.iteration in range(1, self.total_iterations + 1):
            # For GUI debugging, please uncomment the following line. 
            # Note that it doesn't be tested for a long time and may cause some unexpected issues.
            # self._handle_gui()
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
        
        self.gaussians.update_learning_rate(self.iteration)
        self.config.set_debug(True if self.iteration - 1 == self.config.app.debug_from else False)
        losses, timing = phase.on_iteration(self.iteration)
        phase.time += timing
        
        metrics = TrainingMetrics(
            iteration=self.iteration,
            phase=self.cur_phase,
            losses=losses,
            timing=timing
        )
        
        for observer in self.observers:
            observer.on_iteration_end(metrics)
        
    
    def _switch_phase(self, new_phase):
        
        if self.cur_phase != -1:
            self.phases[self.cur_phase].on_phase_end()
            self.meta_information.append((self.phases[self.cur_phase].name, 
                                          self.phases[self.cur_phase].time))
            self.phases[self.cur_phase] = None
        
        last_phase = self.cur_phase
        self.cur_phase = new_phase
        self.phases[self.cur_phase].on_phase_start()
        
        for observer in self.observers:
            observer.on_phase_changed(last_phase, new_phase)        

        
    def _init_components(self):
        self.gaussians = GaussianModel(self.config.model.sh_degree)
        self.scene = Scene(self.config.model, self.gaussians, self.config.ckpt.load_iterations, shuffle=False)
        
        bg_color = [1, 1, 1] if self.config.model.white_background else [0, 0, 0]
        self.background = torch.tensor(bg_color, dtype=torch.float32, device=self.config.model.data_device)
        
        self.first_iter = 1
        if self.config.ckpt.start_checkpoint is not None:
            self._load_checkpoint()

        self.gaussians.training_setup(self.config.opt)
        self.gaussians._features_rest = torch.zeros_like(self.gaussians._features_rest, device=self.device).requires_grad_(False)
        
        if self.config.model.drop_rate > 0:
            pre_n = self.gaussians._xyz.shape[0]
            print(f"Before Prune {pre_n} points")
            ind = torch.randperm(pre_n, device="cuda")[:int(pre_n * self.config.model.drop_rate)]
            mask = torch.zeros(pre_n, dtype=torch.bool, device="cuda")
            mask[ind] = True
            self.gaussians.prune_points(mask)
            print(f"Prune {len(ind)} points, remain {self.gaussians._opacity.shape[0]} points")

            
        
        
    def _init_phases(self):
        
        phase_iter = 1
        phase_uid = 0
        self.phases = []
        self.total_iterations = 0
        
        def _add_phase(phase, phase_name, num_iter, enable_densify):
            if num_iter == 0:
                return
            
            self.total_iterations += num_iter
            
            nonlocal phase_iter, phase_uid
            
            begin_iter = phase_iter
            end_iter = phase_iter + num_iter - 1
            
            self.phases.append(phase(self, phase_uid, phase_name, begin_iter, end_iter, enable_densify))
            
            phase_iter += num_iter
            phase_uid += 1
        
        
        _add_phase(PreProcessPhase, "Pre Process", self.config.style.preprocess_iter, self.config.style.pre_densify)

        for i in range(self.config.style.rounds):
            enable_densify = self.config.style.enable_stylize_densify and (i < self.config.style.rounds // 2)
            phase = PriorPhase if self.config.style.enable_prior else NNFMPhase
            # For style transfer, we find that the prior phase is not very effective, so we directly use NNFMPhase for all rounds. 
            # For texture style transfer, we keep the prior phase as it is important for stylization.
            phase_name = f"Prior {i}" if self.config.style.enable_prior else f"NNFM {i}"
            _add_phase(phase, phase_name, self.config.style.style_iter, enable_densify)
            if self.config.style.enable_geometry_correction:
                _add_phase(CorrectionPhase, f"Correction {i}", self.config.style.correction_iter, False)
            
            if self.config.style.enable_weighted:
                _add_phase(NNFMPhase,  f"NNFM {i}", self.config.style.style_iter, enable_densify)
                if self.config.style.enable_geometry_correction:
                    _add_phase(CorrectionPhase, f"Correction {i}", self.config.style.correction_iter, False)
                
        
        _add_phase(PostProcessPhase, "Post Process", self.config.style.postpreprocess_iter, False)
        
    def _init_observers(self):
        
        self.observers: List[TrainingObserver] = [
            CheckpointSaver(self),
            ProgressTracker(self),
        ]
        
        
    def _get_train_phase(self):
        new_phase = self.cur_phase
        if self.cur_phase == -1:
            return 0
        while self.iteration > self.phases[new_phase].end_iter:
            new_phase += 1
        return new_phase
        
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
                    (self.iteration < int(self.total_iterations)) or not keep_alive
                ):
                    break
            except Exception:
                network_gui.conn = None

