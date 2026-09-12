import json
import logging
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import os
from typing import Dict, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def evaluate_model(model: nn.Module, test_loader: DataLoader) -> Tuple[float, np.ndarray]:
    """
    Evaluate the TinyConvNet model on the test dataset.

    Args:
        model (nn.Module): The TinyConvNet model.
        test_loader (DataLoader): DataLoader for test data.

    Returns:
        Tuple[float, np.ndarray]: A tuple containing accuracy and confusion matrix.
    """
    if len(test_loader) == 0:
        logging.warning('The test_loader is empty. No evaluation will be performed.')
        return 0.0, np.array([])

    try:
        # Set model to evaluation mode
        model.eval()
        all_preds = []
        all_labels = []

        # Check if model is on the correct device
        device = next(model.parameters()).device

        # Iterate over test data
        with torch.no_grad():
            for data, labels in test_loader:
                data, labels = data.to(device), labels.to(device)  # Move data and labels to the same device
                outputs = model(data)
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())  # Move predictions to CPU for numpy conversion
                all_labels.extend(labels.cpu().numpy())  # Move labels to CPU for numpy conversion

        # Compute accuracy
        accuracy = accuracy_score(all_labels, all_preds)
        logging.info(f'Accuracy: {accuracy:.4f}')

        # Calculate confusion matrix
        conf_matrix = confusion_matrix(all_labels, all_preds)
        if conf_matrix.size > 0:
            logging.info(f'Confusion Matrix:\n{conf_matrix}')
        else:
            logging.warning('The confusion matrix is empty. No metrics to log.')

        # Log evaluation metrics
        log_metrics(accuracy, conf_matrix)

        return accuracy, conf_matrix

    except Exception as e:
        logging.error(f'Error during evaluation: {e}')
        raise

def log_metrics(accuracy: float, conf_matrix: np.ndarray):
    """
    Log the evaluation metrics and save results in JSON and CSV formats.

    Args:
        accuracy (float): The accuracy of the model.
        conf_matrix (np.ndarray): The confusion matrix.
    """
    results = {
        'accuracy': accuracy,
        'confusion_matrix': conf_matrix.tolist() if conf_matrix.size > 0 else []  # Convert to list for JSON serialization
    }

    # Save results to JSON
    with open('evaluation_results.json', 'w') as json_file:
        json.dump(results, json_file, indent=4)
    
    # Save results to CSV
    if conf_matrix.size > 0:
        np.savetxt('confusion_matrix.csv', conf_matrix, delimiter=',', header=','.join(map(str, range(conf_matrix.shape[1]))), comments='')

    # Generate visualization for confusion matrix
    visualize_confusion_matrix(conf_matrix)

def visualize_confusion_matrix(conf_matrix: np.ndarray):
    """
    Generate a heatmap for the confusion matrix.

    Args:
        conf_matrix (np.ndarray): The confusion matrix.
    """
    if conf_matrix.size == 0:
        logging.warning('Confusion matrix is empty. Visualization will not be generated.')
        return

    if conf_matrix.shape[0] != conf_matrix.shape[1]:
        logging.warning('Confusion matrix is not square. Visualization will not be generated.')
        return

    plt.figure(figsize=(10, 7))
    plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(conf_matrix.shape[0])
    plt.xticks(tick_marks, tick_marks)
    plt.yticks(tick_marks, tick_marks)

    # Normalize the confusion matrix
    conf_matrix_normalized = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]
    
    thresh = conf_matrix_normalized.max() / 2.
    for i, j in np.ndindex(conf_matrix_normalized.shape):
        plt.text(j, i, f'{conf_matrix_normalized[i, j]:.2f}', horizontalalignment="center",
                 color="white" if conf_matrix_normalized[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.close()