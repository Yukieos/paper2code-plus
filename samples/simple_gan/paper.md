# MiniGAN: A Small Generative Adversarial Network for Digit Image Synthesis

## Abstract

We present MiniGAN, a compact generative adversarial network that
synthesizes 28x28 grayscale digit images from random noise. MiniGAN
consists of a fully-connected generator and a fully-connected
discriminator, trained jointly with the standard adversarial minimax
objective. It serves as a minimal, easy-to-reproduce reference
implementation of the GAN training pattern.

## 1. Introduction

Generative adversarial networks (GANs) train a generator and a
discriminator in opposition: the generator learns to produce realistic
samples, while the discriminator learns to distinguish real samples from
generated ones. This paper describes MiniGAN, a small fully-connected GAN
intended as a clear reference implementation.

## 2. Model Architecture

### 2.1 Generator

The generator maps a latent noise vector to a 28x28 grayscale image:

- **Input**: latent vector z of dimension 64, sampled from a standard
  normal distribution.
- **FC1**: Linear(64, 256) -> LeakyReLU(negative_slope=0.2).
- **FC2**: Linear(256, 512) -> LeakyReLU(negative_slope=0.2).
- **FC3 (output)**: Linear(512, 784) -> Tanh, reshaped to a 1x28x28 image
  with pixel values in [-1, 1].

### 2.2 Discriminator

The discriminator maps a 28x28 image to a real/fake probability:

- **Input**: flattened 784-dim image vector.
- **FC1**: Linear(784, 512) -> LeakyReLU(negative_slope=0.2) ->
  Dropout(p=0.3).
- **FC2**: Linear(512, 256) -> LeakyReLU(negative_slope=0.2) ->
  Dropout(p=0.3).
- **FC3 (output)**: Linear(256, 1) -> Sigmoid, producing a scalar
  probability that the input image is real.

## 3. Dataset

The model is trained on a standard 10-class handwritten-digit dataset of
28x28 grayscale images (60,000 training images). Pixel values are scaled
from [0, 255] to [-1, 1] to match the generator's Tanh output range. Class
labels are not used during GAN training (the generator is unconditional).

## 4. Training Procedure

- **Loss**: standard non-saturating GAN loss. The discriminator minimizes
  binary cross-entropy between its predictions and real/fake labels; the
  generator maximizes the discriminator's predicted probability that
  generated images are real (equivalently, minimizes binary cross-entropy
  against a "real" target on generated images).
- **Optimizer**: Adam for both networks, learning rate 2e-4, betas=(0.5, 0.999).
- **Batch size**: 128.
- **Epochs**: 20.
- **Training loop**: for each batch, first update the discriminator on one
  batch of real images and one batch of freshly generated fake images,
  then update the generator once using the discriminator's response to a
  new batch of fake images.
- A fixed batch of latent vectors is sampled once at the start of training
  and used to generate sample images for qualitative inspection after
  every epoch.

## 5. Evaluation

We report the generator and discriminator loss curves over training, and
Frechet Inception Distance (FID) computed between 10,000 generated images
and 10,000 real held-out images as the primary quantitative metric, along
with a qualitative grid of generated sample images.

## 6. Limitations

MiniGAN uses simple fully-connected layers rather than convolutional
layers, is unconditional (no class control over generated digits), and is
evaluated on a single, simple image domain; results should not be
extrapolated to higher-resolution or more complex image generation tasks.
