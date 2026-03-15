# Geometry-aware Texture Transfer for Gaussian Splatting

[![arXiv](https://img.shields.io/badge/arXiv-2505.15208-b31b1b.svg)](https://www.arxiv.org/abs/2505.15208)
[![Project Page](https://img.shields.io/badge/Project-Website-green)](https://vpx-ecnu.github.io/GT2-GS-website/)

This repository contains the official implementation of the paper **"Geometry-aware Texture Transfer for Gaussian Splatting"**, introducing a novel approach for texture transfer in 3D scenes represented by Gaussian Splatting.


<!-- </video>
![](./assets/fern_banded_0042.mp4)
### Style Transfer
![](./abcgs/assets/semantic_flower.jpg)
### Compositional Style Transfer
![](./abcgs/assets/compositional_fern.jpg) -->

## Update🔥
- [2025/11/08] [GT²-GS](https://vpx-ecnu.github.io/GT2-GS-website/) has been accepted by **AAAI 2026**! Code is coming soon!
- [2025/05/22] We release the paper on [arXiv](https://www.arxiv.org/abs/2505.15208).

## Installation

### Requirements
- NVIDIA GPU with CUDA 11.8
- Python 3.10
- PyTorch 2.3.0

### Conda
```bash
# Clone repository with submodules
git clone https://github.com/vpx-ecnu/GT2-GS --recursive
cd GT2-GS

# Install Python dependencies
conda env create -f environment.yaml
conda activate GT2-GS
pip install gs/submodules/diff-gaussian-rasterization
pip install gs/submodules/simple-knn
```

## Quick Start
### Dataset and Checkpoint
* For scene dataset, you can find LLFF dataset in [NeRF](https://github.com/bmild/nerf) and T&T dataset in [3DGS](https://github.com/graphdeco-inria/gaussian-splatting). 
* For style dataset, you can find it in [here](https://drive.google.com/file/d/10EPUQpH0PE8Mnoxxs1URePtjQZElt--s/view?usp=sharing).
* For texture dataset, you can find it in here. #TODO
* **For optimal stylization results, ensure that the original scene is trained using 0th-order spherical harmonics (SH) coefficients.** Higher-order SH coefficients may introduce artifacts or inconsistencies during the style transfer process. Using 0th-order SH coefficients ensures smoother and more coherent stylization.

### Texture Transfer
```
python style_main.py --config configs/llff_texture.yaml --stylized_model_path ./output/texture/llff/fern
python scripts/render_llff_video.py --config ./output/texture/llff/fern/config.yaml
```
### Style Transfer
```
python style_main.py --config configs/llff_style.yaml --stylized_model_path ./output/llff/fern
python scripts/render_llff_video.py --config ./output/llff/fern/config.yaml
```

Please check `python style_main.py --help` or files under `configs/` for help.

## Contact

If you have any questions or suggestions, feel free to open an issue on GitHub.
You can also contact [Garv1tum](https://github.com/Grav1tum) and [lzlcs](https://github.com/lzlcs) directly.


## Citation

If you find this project useful, please give a star⭐ to this repo and cite our paper:
```bibtex
@article{liu2025gt2,
  title={GT2-GS: Geometry-aware Texture Transfer for Gaussian Splatting},
  author={Liu, Wenjie and Liu, Zhongliang and Shu, Junwei and Wang, Changbo and Li, Yang},
  journal={arXiv preprint arXiv:2505.15208},
  year={2025}
}
```