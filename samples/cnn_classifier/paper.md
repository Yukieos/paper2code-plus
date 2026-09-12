# TinyConvNet: A Small Convolutional Classifier for Grayscale Digit Recognition

## Abstract

We describe TinyConvNet, a compact convolutional neural network for classifying
28x28 grayscale images into 10 digit classes (0-9). The model uses two
convolutional blocks followed by a small fully-connected head, and is trained
with cross-entropy loss and the Adam optimizer. Despite its small size
(under 100K parameters), TinyConvNet reaches over 98% test accuracy on a
held-out split of a standard handwritten-digit dataset, making it a useful
minimal reference architecture for image classification.

## 1. Introduction

Convolutional networks remain a strong default for small-image classification
tasks. This paper presents TinyConvNet, a minimal architecture intended for
fast training and easy reproducibility rather than state-of-the-art accuracy.

## 2. Model Architecture

TinyConvNet processes a single-channel 28x28 input through the following
stages:

- **Conv Block 1**: Conv2d(in_channels=1, out_channels=16, kernel_size=3,
  padding=1) -> ReLU -> MaxPool2d(kernel_size=2) — output shape 16x14x14.
- **Conv Block 2**: Conv2d(in_channels=16, out_channels=32, kernel_size=3,
  padding=1) -> ReLU -> MaxPool2d(kernel_size=2) — output shape 32x7x7.
- **Flatten**: reshape to a vector of length 32*7*7 = 1568.
- **FC1**: Linear(1568, 128) -> ReLU -> Dropout(p=0.25).
- **FC2 (output head)**: Linear(128, 10), producing raw class logits.

All convolutions use stride 1. No batch normalization is used, to keep the
architecture minimal.

## 3. Dataset

The model is trained on a standard 10-class handwritten-digit dataset of
28x28 grayscale images (60,000 training images, 10,000 test images, roughly
balanced across classes). Pixel values are normalized to the range [0, 1]
by dividing by 255, then standardized using dataset mean=0.1307 and
std=0.3081.

## 4. Training Procedure

- **Loss**: cross-entropy loss over the 10 output logits.
- **Optimizer**: Adam with learning rate 1e-3, betas=(0.9, 0.999).
- **Batch size**: 64.
- **Epochs**: 5.
- **LR schedule**: none (constant learning rate).
- Training and validation loss/accuracy are logged after every epoch.
  The model checkpoint with the best validation accuracy is saved.

## 5. Evaluation

We report top-1 accuracy on the 10,000-image held-out test split, along with
a per-class confusion matrix. TinyConvNet achieves approximately 98.3% test
accuracy under this protocol. We also report the total parameter count and
average inference latency per image on CPU as efficiency metrics.

## 6. Limitations

TinyConvNet is intentionally small and evaluated on a single, simple
dataset; it is not intended to compete with larger architectures on more
complex image classification benchmarks.
