# MiniEncoder: A Small Transformer Encoder for Sentiment Classification

## Abstract

We present MiniEncoder, a compact Transformer-encoder model for binary
sentiment classification of short text sequences. The model embeds input
tokens, adds sinusoidal positional encodings, applies two Transformer
encoder layers with multi-head self-attention, and pools the sequence into
a single label via mean pooling followed by a linear classification head.
MiniEncoder is designed as a minimal, easy-to-reproduce reference
implementation of the encoder-only Transformer pattern.

## 1. Introduction

Self-attention-based encoders are the standard building block for modern
text classification. This paper describes a deliberately small
encoder-only model, MiniEncoder, suitable as a reference implementation and
teaching example rather than a benchmark-leading system.

## 2. Model Architecture

Given an input sequence of token ids of length up to 128:

- **Token Embedding**: Embedding(vocab_size=20000, embedding_dim=128).
- **Positional Encoding**: fixed sinusoidal positional encodings of
  dimension 128, added elementwise to the token embeddings.
- **Encoder Layer x2**: each layer consists of:
  - Multi-head self-attention with num_heads=4, embedding_dim=128
    (head_dim=32), followed by a residual connection and LayerNorm.
  - A position-wise feed-forward block: Linear(128, 256) -> ReLU ->
    Linear(256, 128), followed by a residual connection and LayerNorm.
  - Dropout with p=0.1 applied after attention and after the feed-forward
    block.
- **Pooling**: mean-pool the final layer's token representations over the
  (non-padding) sequence positions to obtain a single 128-dim vector per
  example.
- **Classification Head**: Linear(128, 2), producing logits for the two
  sentiment classes (negative, positive).

## 3. Dataset

The model is trained on a binary sentiment classification dataset of short
movie-review-style text snippets, split into 25,000 training and 25,000
test examples, roughly balanced between the two classes. Text is
lowercased, tokenized with a simple whitespace/punctuation tokenizer, and
mapped to ids using a vocabulary of the 20,000 most frequent tokens (rare
tokens map to an <unk> id). Sequences are truncated or padded to length
128 with a <pad> id.

## 4. Training Procedure

- **Loss**: cross-entropy loss over the 2 output logits.
- **Optimizer**: Adam with learning rate 3e-4, weight decay 0.01.
- **Batch size**: 32.
- **Epochs**: 3.
- **LR schedule**: linear warmup over the first 10% of training steps,
  then linear decay to 0.
- Padding positions are masked out of the self-attention computation and
  excluded from the mean-pooling step.

## 5. Evaluation

We report classification accuracy and F1 score on the held-out test split.
MiniEncoder reaches approximately 87% test accuracy under this protocol. We
additionally report the total parameter count as an efficiency metric.

## 6. Limitations

MiniEncoder uses only 2 encoder layers and a small embedding dimension, and
is evaluated on a single binary classification dataset; results should not
be extrapolated to larger-scale language understanding tasks.
