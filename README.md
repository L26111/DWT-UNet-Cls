# DWT-UNet-Cls
PaddlePaddle implementation of "Enhancing Pavement Disease Classification through Wavelet Frequency Features and Attention Mechanisms"

## Requirements
- paddlepaddle
- numpy  
- opencv-python
- Pillow

## Files
- model.py: Main DWT-UNet model implementation
- train.py: Training script for the main model
- model_csp_newwave.py: Alternative model variant
- train_csp_newwave.py: Training script for variant model
- dataset.py: Dataset loading and preprocessing
- best_model.pdparams: Pre-trained model weights

## Quick Start
1. Install dependencies: pip install -r requirements.txt
2. Prepare your dataset
3. Run training: python train.py

## Pre-trained Model
Due to GitHub file size limitations, the pre-trained model is available at: 
[Download Link](https://pan.baidu.com/s/1WusD7qQUqvfrO-gO_GAfag?pwd=1234 ) (extraction code: 1234)

## Citation  
If you use this code or the pretrained weights, please cite our paper:  
> Qian Liu et al., "Enhancing Pavement Disease Classification through Wavelet Frequency Features and Attention Mechanisms," *The Visual Computer*, 2025 (under review).
