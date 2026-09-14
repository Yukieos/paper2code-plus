# BoWLogReg: Bag-of-Words Logistic Regression for Binary Sentiment Classification

## Abstract

We describe BoWLogReg, a minimal logistic-regression classifier that predicts
binary sentiment (positive/negative) from short text reviews represented as
bag-of-words count vectors. The model is a single linear layer followed by a
sigmoid, trained with binary cross-entropy and stochastic gradient descent. It
serves as a minimal, reproducible reference for text classification without
neural sequence models.

## 1. Introduction

Linear models over bag-of-words features remain a competitive, interpretable
baseline for short-text sentiment classification. This paper presents
BoWLogReg, a deliberately simple architecture intended for fast training and
easy reproduction.

## 2. Feature Representation

Each input review is lowercased, tokenized on whitespace and punctuation, and
mapped to a fixed vocabulary of the `V` most frequent training tokens
(default `V = 20000`). A review becomes a length-`V` vector of token counts.
Out-of-vocabulary tokens are ignored. Count vectors are optionally scaled by
inverse document frequency (TF-IDF).

## 3. Model Architecture

- **Input**: a length-`V` bag-of-words (or TF-IDF) vector.
- **Linear layer**: Linear(V, 1), producing a single logit.
- **Activation**: sigmoid, producing a probability of the positive class.

The model has `V + 1` trainable parameters (weights plus bias).

## 4. Dataset

The model is trained on a standard binary sentiment dataset of short reviews
(for example, 25,000 training reviews and 25,000 test reviews, balanced across
the two classes). The vocabulary is built from the training split only. Data
is split 80/20 into train/validation from the training portion; the official
test split is held out for final evaluation.

## 5. Training Procedure

- **Loss**: binary cross-entropy between the predicted probability and the
  0/1 label.
- **Optimizer**: stochastic gradient descent (SGD) with learning rate 0.1 and
  momentum 0.9.
- **Regularization**: L2 weight decay of 1e-4.
- **Batch size**: 256.
- **Epochs**: 10.
- Training and validation loss and accuracy are logged after every epoch. The
  checkpoint with the highest validation accuracy is saved.

## 6. Evaluation

We report accuracy, precision, recall, and F1 score on the held-out test
split, along with the area under the ROC curve (AUC).

## 7. Limitations

BoWLogReg ignores word order and cannot capture negation or long-range
context; it is intended as a simple linear baseline only.
