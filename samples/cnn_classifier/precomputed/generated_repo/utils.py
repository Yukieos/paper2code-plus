import os
import json
import logging
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def log_metrics(epoch: int, loss: float, accuracy: float) -> None:
    """
    Log training metrics to console or file.

    Args:
        epoch (int): Current epoch number.
        loss (float): Current loss value.
        accuracy (float): Current accuracy value.
    """
    logging.info(f'Epoch: {epoch}, Loss: {loss:.4f}, Accuracy: {accuracy:.4f}')

def calculate_accuracy(predictions: np.ndarray, labels: np.ndarray) -> float:
    """
    Calculate accuracy given predictions and true labels.

    Args:
        predictions (np.ndarray): Model predictions.
        labels (np.ndarray): True labels.

    Returns:
        float: Accuracy as a percentage.
    """
    correct_predictions = np.sum(predictions == labels)
    accuracy = correct_predictions / len(labels)
    return accuracy

def save_results(results: Dict, file_path: str) -> None:
    """
    Save evaluation results in JSON format.

    Args:
        results (Dict): Results to save.
        file_path (str): Path to save the results.
    """
    try:
        with open(file_path, 'w') as f:
            json.dump(results, f, indent=4)
        logging.info(f'Results saved to {file_path}')
    except Exception as e:
        logging.error(f'Error saving results: {e}')

def visualize_metrics(losses: List[float], accuracies: List[float]) -> None:
    """
    Visualize training metrics over epochs.

    Args:
        losses (List[float]): List of loss values over epochs.
        accuracies (List[float]): List of accuracy values over epochs.
    """
    try:
        epochs = range(1, len(losses) + 1)

        plt.figure(figsize=(12, 5))

        # Plot loss
        plt.subplot(1, 2, 1)
        plt.plot(epochs, losses, label='Loss', color='blue')
        plt.title('Loss over Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()

        # Plot accuracy
        plt.subplot(1, 2, 2)
        plt.plot(epochs, accuracies, label='Accuracy', color='green')
        plt.title('Accuracy over Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()

        plt.tight_layout()
        plt.show()
    except Exception as e:
        logging.error(f'Error visualizing metrics: {e}')

def load_checkpoint(checkpoint_path: str):
    """
    Load model checkpoint.

    Args:
        checkpoint_path (str): Path to the checkpoint file.

    Returns:
        dict: Loaded checkpoint data.
    """
    if not os.path.exists(checkpoint_path):
        logging.error(f'Checkpoint file not found: {checkpoint_path}')
        return None

    try:
        checkpoint = torch.load(checkpoint_path)
        logging.info(f'Checkpoint loaded from {checkpoint_path}')
        return checkpoint
    except Exception as e:
        logging.error(f'Error loading checkpoint: {e}')
        return None

def evaluate_model(model, data_loader, device) -> Dict:
    """
    Evaluate the model on the validation/test dataset.

    Args:
        model: The model to evaluate.
        data_loader: DataLoader for the dataset.
        device: Device to run the evaluation on.

    Returns:
        Dict: Evaluation results including accuracy and loss.
    """
    model.eval()
    total_loss = 0
    correct_predictions = 0
    total_samples = 0

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            correct_predictions += (predicted == labels).sum().item()
            total_samples += labels.size(0)

    average_loss = total_loss / len(data_loader)
    accuracy = correct_predictions / total_samples

    results = {
        'loss': average_loss,
        'accuracy': accuracy
    }
    logging.info(f'Evaluation results: {results}')
    return results