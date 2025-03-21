# TiPGAN: High-Quality Tileable Textures Synthesis with Intrinsic Priors for Cloth Digitization Applications
This is the official implementation of TiPGAN (CAD 2025). If you find our repository is helpful, please cite this [bib](#Citation).
[![TiPGAN Demo](https://github.com/user-attachments/assets/66a173c9-a8ad-4cf6-8762-b744e186eded)](https://youtu.be/Tyk7mGeElzg)

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

# Citation

If this repository is helpful to your research, please cite it as below.

```bibtex
@article{HE2025103866,
title = {TiPGAN: High-quality tileable textures synthesis with intrinsic priors for cloth digitization applications},
journal = {Computer-Aided Design},
volume = {183},
pages = {103866},
year = {2025},
issn = {0010-4485},
doi = {https://doi.org/10.1016/j.cad.2025.103866},
url = {https://www.sciencedirect.com/science/article/pii/S0010448525000284},
author = {Honghong He and Zhengwentai Sun and Jintu Fan and P.Y. Mok},
}

```
