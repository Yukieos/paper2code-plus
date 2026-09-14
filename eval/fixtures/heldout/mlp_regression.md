# TabMLP: A Small Multilayer Perceptron for Tabular Regression

## Abstract

We describe TabMLP, a compact fully-connected neural network for predicting a
single continuous target from fixed-length tabular feature vectors. The model
is a three-layer multilayer perceptron trained with mean squared error and the
Adam optimizer. TabMLP is intended as a minimal, easily reproducible reference
architecture for supervised regression on standardized numeric features.

## 1. Introduction

Multilayer perceptrons remain a strong baseline for tabular regression when
features are already numeric and of moderate dimensionality. This paper
presents TabMLP, a deliberately small architecture intended for fast training
and reproducibility rather than state-of-the-art accuracy.

## 2. Model Architecture

TabMLP maps an input feature vector of dimension `d_in` to a scalar prediction
through the following stages:

- **Input**: a real-valued vector of length `d_in` (default `d_in = 8`).
- **Hidden layer 1**: Linear(d_in, 64) -> ReLU.
- **Hidden layer 2**: Linear(64, 32) -> ReLU -> Dropout(p=0.1).
- **Output head**: Linear(32, 1), producing a single raw regression value.

No batch normalization is used, to keep the architecture minimal. All hidden
activations are ReLU.

## 3. Dataset

The model is trained on a standard tabular regression dataset of numeric
features with a single continuous target (for example, a housing-price style
dataset with roughly 20,000 rows). Features are standardized to zero mean and
unit variance using statistics computed on the training split only; the same
statistics are applied to the validation and test splits. Data is split
70/15/15 into train/validation/test.

## 4. Training Procedure

- **Loss**: mean squared error (MSE) between the predicted scalar and target.
- **Optimizer**: Adam with learning rate 1e-3, betas=(0.9, 0.999).
- **Batch size**: 128.
- **Epochs**: 30.
- **LR schedule**: reduce learning rate by a factor of 0.5 when validation loss
  plateaus for 3 consecutive epochs.
- Training and validation MSE are logged after every epoch. The checkpoint with
  the lowest validation MSE is saved.

## 5. Evaluation

We report root mean squared error (RMSE) and the coefficient of determination
(R^2) on the held-out test split. We also report the total parameter count as
an efficiency metric.

## 6. Limitations

TabMLP assumes purely numeric, fixed-length inputs and does not handle
categorical features, missing values, or sequence data.
