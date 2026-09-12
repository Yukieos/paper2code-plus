import torch
import torch.nn as nn
import torch.nn.functional as F

class MiniEncoder(nn.Module):
    """
    MiniEncoder model for binary sentiment classification.

    This model implements a compact Transformer encoder architecture designed for 
    classifying sentiment in short text sequences. It consists of an embedding layer, 
    positional encoding, and multiple Transformer encoder layers.

    Attributes:
        vocab_size (int): Size of the vocabulary.
        embedding_dim (int): Dimension of the embeddings.
        num_heads (int): Number of attention heads.
        head_dim (int): Dimension of each attention head.
        dropout_rate (float): Dropout rate for the model.
        embedding (nn.Embedding): Embedding layer for input tokens.
        positional_encoding (nn.Parameter): Positional encoding for the input embeddings.
        transformer_encoder (nn.TransformerEncoder): Transformer encoder layers.
        dropout (nn.Dropout): Dropout layer for regularization.
        classification_head (nn.Linear): Final linear layer for classification.
    """

    def __init__(self, vocab_size: int, embedding_dim: int, num_heads: int, head_dim: int, dropout_rate: float) -> None:
        """
        Initialize the MiniEncoder model with the specified parameters.

        Args:
            vocab_size (int): Size of the vocabulary.
            embedding_dim (int): Dimension of the embeddings.
            num_heads (int): Number of attention heads.
            head_dim (int): Dimension of each attention head.
            dropout_rate (float): Dropout rate for the model.
        """
        super(MiniEncoder, self).__init__()

        # Input validation
        if vocab_size <= 0:
            raise ValueError("vocab_size must be a positive integer.")
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be a positive integer.")
        if num_heads <= 0:
            raise ValueError("num_heads must be a positive integer.")
        if head_dim <= 0:
            raise ValueError("head_dim must be a positive integer.")
        if not (0 <= dropout_rate < 1):
            raise ValueError("dropout_rate must be in the range [0, 1).")

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.dropout_rate = dropout_rate

        # Initialize layers
        self.embedding = nn.Embedding(vocab_size, embedding_dim)  # Embedding layer
        self.positional_encoding = nn.Parameter(torch.zeros(1, 128, embedding_dim))  # Positional encoding
        encoder_layers = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=num_heads, dim_feedforward=embedding_dim * 4, dropout=dropout_rate)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=2)  # Two layers as per the paper
        self.dropout = nn.Dropout(dropout_rate)  # Dropout layer
        self.classification_head = nn.Linear(embedding_dim, 2)  # Output layer for binary classification

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the MiniEncoder model.

        Args:
            x (torch.Tensor): Input tensor of token ids.

        Returns:
            torch.Tensor: Output logits for classification.
        """
        # Input validation
        if not isinstance(x, torch.Tensor):
            raise TypeError("Input x must be a torch.Tensor.")
        if x.dim() != 2:
            raise ValueError("Input x must be a 2D tensor of shape (batch_size, sequence_length).")

        # Step 1: Pass the input tensor through the embedding layer
        embeddings = self.embedding(x)  # Shape: (batch_size, sequence_length, embedding_dim)

        # Step 2: Add positional encodings to the embeddings
        embeddings += self.positional_encoding[:, :embeddings.size(1), :]  # Broadcasting positional encodings

        # Step 3: Apply the Transformer encoder layers to the embeddings
        # Reshape for transformer encoder: (sequence_length, batch_size, embedding_dim)
        embeddings = embeddings.permute(1, 0, 2)
        transformer_output = self.transformer_encoder(embeddings)  # Shape: (sequence_length, batch_size, embedding_dim)

        # Step 4: Pool the output and pass it through the classification head
        pooled_output = torch.mean(transformer_output, dim=0)  # Mean pooling over the sequence length
        logits = self.classification_head(self.dropout(pooled_output))  # Shape: (batch_size, 2)

        return logits  # Output logits for classification