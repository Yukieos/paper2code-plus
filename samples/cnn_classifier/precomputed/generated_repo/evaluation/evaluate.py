import json
import logging
import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix
from torch.utils.data import DataLoader
from typing import Tuple

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def evaluate_model(model: torch.nn.Module, test_loader: DataLoader) -> Tuple[float, np.ndarray]:
    """
    Evaluates the TinyConvNet model on the test dataset and returns accuracy and confusion matrix,
    with error handling for evaluation process.

    Args:
        model (torch.nn.Module): The TinyConvNet model instance.
        test_loader (DataLoader): Data loader for testing data.

    Returns:
        Tuple[float, np.ndarray]: Accuracy of the model on the test dataset and confusion matrix.
    """
    try:
        # Set model to evaluation mode
        model.eval()
        all_preds = []
        all_labels = []

        # Iterate over test_loader
        with torch.no_grad():
            for data in test_loader:
                inputs, labels = data
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        # Calculate accuracy and confusion matrix
        accuracy = accuracy_score(all_labels, all_preds)
        conf_matrix = confusion_matrix(all_labels, all_preds)

        # Log evaluation results
        logger.info(f"Evaluation completed. Accuracy: {accuracy:.4f}")
        logger.info(f"Confusion Matrix:\n{conf_matrix}")

        # Save results in structured format
        results = {
            "accuracy": accuracy,
            "confusion_matrix": conf_matrix.tolist()  # Convert to list for JSON serialization
        }
        with open('evaluation_results.json', 'w') as f:
            json.dump(results, f, indent=4)

        return accuracy, conf_matrix

    except Exception as e:
        logger.error(f"Error during evaluation: {e}")
        raise e  # Re-raise the exception after logging

# Additional functions for visualization and other metrics can be added here as needed.