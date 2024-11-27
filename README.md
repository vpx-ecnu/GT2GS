```bash
# inside docker
apt -y update
apt install -y git curl wget libgl1-mesa-glx libglib2.0-0 libx11-6

conda create --name ABCGS python=3.10 -y
conda activate ABCGS
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y
pip install submodules/lang-segment-anything
pip install submodules/diff-gaussian-rasterization
pip install submodules/simple-knn
pip install imageio wandb plyfile open3d wandb
```


```bash
bash scripts/single_transfer.sh [scene name] [style image] [GPU ID] [Port]
bash scripts/single_prompt_transfer.sh [scene name] [style image] [scene prompt] [style prompt] [GPU ID] [Port]
bash scripts/multiple_transfer.sh [scene name] [style image1] [style image2] [scene prompt] [GPU ID] [Port]
```