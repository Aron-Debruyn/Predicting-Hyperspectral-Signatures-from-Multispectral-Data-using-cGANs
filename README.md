# Predicting Hyperspectral Signatures from Multispectral Data using cGANs for Agricultural Applications
**Master's thesis project — Aron Debruyn**  
_Reconstruct hyperspectral (476–914 nm) signatures from multispectral, RGB and two band scenarios inputs using conditional GANs._ 

[//]: # "[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14889285.svg)](https://doi.org/10.5281/zenodo.14889285)"
---

## Abstract
This repository contains code, model definitions and utilities developed for the publication **"Predicting Hyperspectral Signatures from Multispectral Data using cGANs for Agricultural Applications"**. The project investigates multiple cGAN generator and discriminator architectures (CNN, U-Net, ResNet, Cramér, MLP and PatchGAN, Cramér and MLP discriminators) and several input-band scenarios (RGB, red-edge, NIR, MI/PCA selected bands) to reconstruct hyperspectral signatures in the 476–914 nm range for agricultural monitoring. The study shows ResNet-based models are generally most stable and accurate, and including red-edge/NIR bands strongly improves performance.


---

## Requirements
- Python 3.11
- PyTorch 2.6.0
- NumPy 2.0
- SciPy 1.13.0
- Matplotlib 3.10
  
---

## Architectures

### CNN Generator
![CNN architecture](/figures/network_cnn_page-0001.jpg)

### U-Net Generator
![U-Net architecture](/figures/network_unet_page-0001.jpg)

### ResNet Generator
![ResNet architecture](/figures/ResNet_page-0001.jpg)

### MLP Generator
![MLP generator architecture](/figures/MLP_page-0001.jpg)

### Cramér Generator
![Cramer generator architecture](/figures/Cramer.JPG)

### PatchGAN Discriminator
![PatchGAN discriminator architecture](/figures/network_dis_fr_page-0001.jpg)

### MLP Discriminator
![MLP discriminator architecture](/figures/MLP_dis_page-0001.jpg)

### Cramér Discriminator
![Cramer discriminator architecture](/figures/Large_dis_page-0001.jpg)
