# TiPGAN: High-Quality Tileable Textures Synthesis with Intrinsic Priors for Cloth Digitization Applications
[![TiPGAN Demo](https://github.com/user-attachments/assets/66a173c9-a8ad-4cf6-8762-b744e186eded)](https://youtu.be/x)

# Environment

We tested our environment on CUDA 11.8. This repository should not have specific requirements on CUDA or PyTorch version. You may should make sure the CUDA version and the PyTorch version are compatible.

```bash
conda create -n tipgan python=3.10
conda activate tipgan
pip install torch==2.4.1 torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cu118
```

Install the requirements.txt:

```bash
pip install -r requirements.txt
```

# Train
```bash
python train_tipgan.py --img_path media/nature_0001.jpg
```

# Inference
```bash
python inference.py
```
