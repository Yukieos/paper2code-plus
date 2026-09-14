# TinyCAE: A Convolutional Autoencoder for Grayscale Image Reconstruction

## Abstract

We describe TinyCAE, a compact convolutional autoencoder that compresses
28x28 grayscale images into a low-dimensional latent code and reconstructs
them. The model is trained with mean squared reconstruction error and the Adam
optimizer. TinyCAE is intended as a minimal, reproducible reference for
unsupervised representation learning on small images.

## 1. Introduction

Convolutional autoencoders are a standard approach for unsupervised feature
learning and denoising on images. This paper presents TinyCAE, a deliberately
small architecture intended for fast training and easy reproduction rather
than state-of-the-art reconstruction quality.

## 2. Model Architecture

TinyCAE has a symmetric encoder-decoder structure operating on single-channel
28x28 inputs.

**Encoder:**
- Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=2, padding=1)
  -> ReLU — output 16x14x14.
- Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1)
  -> ReLU — output 32x7x7.
- Flatten and Linear(32*7*7, 64) to produce the latent code of dimension 64.

**Decoder:**
- Linear(64, 32*7*7) -> ReLU, reshaped to 32x7x7.
- ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1)
  -> ReLU — output 16x14x14.
- ConvTranspose2d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=1)
  -> Sigmoid — output 1x28x28 in the range [0, 1].

## 3. Dataset

The model is trained on a standard 10-class handwritten-digit dataset of
28x28 grayscale images (60,000 training images, 10,000 test images). Labels
are ignored; only the images are used. Pixel values are normalized to the
range [0, 1] by dividing by 255.

## 4. Training Procedure

- **Loss**: mean squared error between the reconstructed and input images.
- **Optimizer**: Adam with learning rate 1e-3.
- **Batch size**: 128.
- **Epochs**: 20.
- **LR schedule**: none (constant learning rate).
- Training and validation reconstruction loss are logged after every epoch.
  The checkpoint with the lowest validation loss is saved.

## 5. Evaluation

We report mean per-pixel MSE and peak signal-to-noise ratio (PSNR) on the
held-out test split. We also visualize a grid of original versus reconstructed
images for qualitative inspection.

## 6. Limitations

TinyCAE targets a single simple dataset and a small latent dimension; it is
not intended for high-resolution or color images.
