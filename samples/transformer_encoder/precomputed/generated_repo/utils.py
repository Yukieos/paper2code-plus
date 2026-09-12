import numpy as np
import torch
import torch.nn.functional as F
from typing import List, Tuple

def validate_input(data: List[int], vocab_size: int) -> None:
    """
    Validate the input data to ensure it contains valid token ids.

    Args:
        data (List[int]): The input sequence of token ids.
        vocab_size (int): The size of the vocabulary.

    Raises:
        ValueError: If any token id is out of the valid range.
    """
    if not all(0 <= token_id < vocab_size for token_id in data):
        raise ValueError(f"All token ids must be in the range [0, {vocab_size}).")


def mean_pooling(embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """
    Perform mean pooling on the embeddings based on the attention mask.

    Args:
        embeddings (torch.Tensor): The embeddings from the transformer model.
        attention_mask (torch.Tensor): The attention mask indicating valid tokens.

    Returns:
        torch.Tensor: The pooled representation of the input sequence.
    """
    # Calculate the sum of embeddings where attention mask is 1
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(embeddings.size())
    sum_embeddings = torch.sum(embeddings * input_mask_expanded, 1)  # Equation (1)
    
    # Count the number of valid tokens
    sum_mask = input_mask_expanded.sum(1)  # Equation (2)
    
    # Avoid division by zero
    sum_mask = torch.clamp(sum_mask, min=1e-9)  # Prevent division by zero
    pooled_output = sum_embeddings / sum_mask  # Equation (3)
    
    return pooled_output


def compute_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    Compute the cross-entropy loss for the model predictions.

    Args:
        logits (torch.Tensor): The output logits from the model.
        labels (torch.Tensor): The true labels for the input data.

    Returns:
        torch.Tensor: The computed loss value.
    """
    # Using F.cross_entropy which combines softmax and negative log likelihood
    loss = F.cross_entropy(logits, labels)  # Equation (4)
    return loss


def accuracy(predictions: torch.Tensor, labels: torch.Tensor) -> float:
    """
    Calculate the accuracy of the model predictions.

    Args:
        predictions (torch.Tensor): The predicted logits from the model.
        labels (torch.Tensor): The true labels for the input data.

    Returns:
        float: The accuracy as a percentage.
    """
    preds = torch.argmax(predictions, dim=1)  # Get the predicted class
    correct = (preds == labels).sum().item()  # Count correct predictions
    acc = correct / labels.size(0)  # Calculate accuracy
    return acc


def f1_score(predictions: torch.Tensor, labels: torch.Tensor) -> float:
    """
    Calculate the F1 score for the model predictions.

    Args:
        predictions (torch.Tensor): The predicted logits from the model.
        labels (torch.Tensor): The true labels for the input data.

    Returns:
        float: The F1 score.
    """
    preds = torch.argmax(predictions, dim=1)  # Get the predicted class
    tp = ((preds == 1) & (labels == 1)).sum().item()  # True Positives
    fp = ((preds == 1) & (labels == 0)).sum().item()  # False Positives
    fn = ((preds == 0) & (labels == 1)).sum().item()  # False Negatives

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0  # Equation (5)
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # Equation (6)

    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0  # Equation (7)
    return f1


def prepare_data(data: List[int], vocab_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Prepare the input data for the model by validating and converting to tensors.

    Args:
        data (List[int]): The input sequence of token ids.
        vocab_size (int): The size of the vocabulary.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing the input tensor and attention mask.
    """
    validate_input(data, vocab_size)  # Validate input
    input_tensor = torch.tensor(data, dtype=torch.long).unsqueeze(0)  # Add batch dimension
    attention_mask = (input_tensor != 0).long()  # Create attention mask (0 for padding)
    return input_tensor, attention_mask