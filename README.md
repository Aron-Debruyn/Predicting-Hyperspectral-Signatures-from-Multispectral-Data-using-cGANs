# Predicting Hyperspectral Signatures from Multispectral Data using cGANs for Agricultural Applications
**Implementation of paper Predicting Hyperspectral Signatures from Multispectral Data using cGANs for Agricultural Applications**
_Reconstruct hyperspectral (476–914 nm) signatures from multispectral, RGB and two band scenarios inputs using conditional GANs._ 


The full dataset used for this study is available on **Zenodo**: 
Here comes the DOI: [//]: # "[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14889285.svg)](https://doi.org/10.5281/zenodo.14889285)"
---

## Abstract
This repository contains code, model definitions and utilities developed for the publication **"Predicting Hyperspectral Signatures from Multispectral Data using cGANs for Agricultural Applications"**. The project investigates multiple cGAN generator and discriminator architectures (CNN, U-Net, ResNet, Cramér, MLP and PatchGAN, Cramér and MLP discriminators) and several input-band scenarios (RGB, red-edge, NIR, MI/PCA selected bands) to reconstruct hyperspectral signatures in the 476–914 nm range for agricultural monitoring. The study shows ResNet-based models are generally most stable and accurate, and including red-edge/NIR bands strongly improves performance.

## Installation & runnning the code
The code is available as a Jupyter Notebook with the name "predicting_hsi_cgans". If you want to reconstruct the findings of our study download the Zenodo "T_data" Numpy files and set the variable "concatenate_larger" to "True" and change and the variable "basedir" in the hyperparameters dictionary to the directory of the Numpy files, additionally in this dictionary you can adjust different training parameters. If you want to use your own original images, take a look at "directory_of_original_images", "subfolders", "load_orig_images", "load_all_images", "concatenate" and "concatenate_larger" and adapt these accordingly. Regardless of which format you used, the training data will be saved in Numpy Memory-mapped files.

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
