import json
import logging
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, confusion_matrix
from torch.utils.data import DataLoader
from typing import Tuple

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def evaluate_model(model: torch.nn.Module, test_loader: DataLoader) -> Tuple[float, np.ndarray]:
    """
    Evaluates the TinyConvNet model on the test dataset and returns accuracy and confusion matrix.

    Args:
        model (nn.Module): The TinyConvNet model to be evaluated.
        test_loader (DataLoader): DataLoader for test data.

    Returns:
        Tuple[float, np.ndarray]: A tuple containing the accuracy and confusion matrix.
    """
    try:
        # Set model to evaluation mode
        model.eval()
        all_labels = []
        all_predictions = []

        # Iterate over test_loader
        with torch.no_grad():
            for inputs, labels in test_loader:
                # Perform forward pass
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)

                # Collect labels and predictions
                all_labels.extend(labels.numpy())
                all_predictions.extend(predicted.numpy())

        # Compute accuracy
        accuracy = accuracy_score(all_labels, all_predictions)
        # Calculate confusion matrix
        conf_matrix = confusion_matrix(all_labels, all_predictions)

        logging.info(f"Model evaluation completed. Accuracy: {accuracy:.4f}")
        return accuracy, conf_matrix

    except Exception as e:
        logging.error(f"Error during model evaluation: {str(e)}")
        raise

def save_results(accuracy: float, conf_matrix: np.ndarray, output_file: str):
    """
    Saves the evaluation results to a structured format (JSON, CSV).

    Args:
        accuracy (float): The accuracy of the model.
        conf_matrix (np.ndarray): The confusion matrix.
        output_file (str): The file path to save the results.
    """
    try:
        # Save accuracy and confusion matrix to JSON
        results = {
            "accuracy": accuracy,
            "confusion_matrix": conf_matrix.tolist()  # Convert numpy array to list for JSON serialization
        }
        with open(output_file, 'w') as json_file:
            json.dump(results, json_file)
        logging.info(f"Results saved to {output_file}")

        # Save confusion matrix to CSV
        conf_matrix_df = pd.DataFrame(conf_matrix)
        conf_matrix_csv_file = output_file.replace('.json', '_confusion_matrix.csv')
        conf_matrix_df.to_csv(conf_matrix_csv_file, index=False)
        logging.info(f"Confusion matrix saved to {conf_matrix_csv_file}")

    except Exception as e:
        logging.error(f"Error saving results: {str(e)}")
        raise

# Example usage
if __name__ == "__main__":
    # Load your model and test_loader here
    # model = ...
    # test_loader = ...
    
    # Evaluate the model
    # accuracy, conf_matrix = evaluate_model(model, test_loader)
    
    # Save the results
    # save_results(accuracy, conf_matrix, 'evaluation_results.json')