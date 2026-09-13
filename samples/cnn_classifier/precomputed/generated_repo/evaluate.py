import torch
import numpy as np
import logging
import json
import os
from sklearn.metrics import accuracy_score, confusion_matrix
from torch.utils.data import DataLoader
from typing import Tuple

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def evaluate(model: torch.nn.Module, test_loader: DataLoader) -> Tuple[float, np.ndarray]:
    """
    Evaluates the TinyConvNet model on the test dataset and returns accuracy and confusion matrix,
    with error handling and logging.

    Args:
        model (torch.nn.Module): The TinyConvNet model instance.
        test_loader (DataLoader): DataLoader for test data.

    Returns:
        Tuple[float, np.ndarray]: Accuracy of the model on test data and confusion matrix.
    """
    try:
        # Set model to evaluation mode
        model.eval()
        all_predictions = []
        all_labels = []

        # Iterate over batches in test_loader
        with torch.no_grad():
            for inputs, labels in test_loader:
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        # Calculate accuracy and confusion matrix
        accuracy = accuracy_score(all_labels, all_predictions)
        conf_matrix = confusion_matrix(all_labels, all_predictions)

        # Log evaluation metrics
        logger.info(f"Accuracy: {accuracy:.4f}")
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
        logger.error(f"Error during evaluation: {str(e)}")
        raise

# Example usage (commented out to avoid execution in module context)
# if __name__ == "__main__":
#     # Load your model and test_loader here
#     model = ...  # Load your trained model
#     test_loader = ...  # Load your test DataLoader
#     evaluate(model, test_loader)