import json
import logging
import os
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def evaluate(model: nn.Module, test_loader: DataLoader) -> dict:
    """
    Evaluate the MiniEncoder model on the test dataset.

    Args:
        model (nn.Module): The trained MiniEncoder model.
        test_loader (DataLoader): DataLoader for the test dataset.

    Returns:
        Dict[str, float]: A dictionary containing evaluation metrics such as accuracy and F1 score.
    """
    try:
        model.eval()  # Set the model to evaluation mode
        all_preds = []
        all_labels = []

        with torch.no_grad():  # Disable gradient calculation
            for inputs, labels in test_loader:
                outputs = model(inputs)  # Forward pass
                _, preds = torch.max(outputs, dim=1)  # Get predictions
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='weighted')

        metrics = {
            'accuracy': accuracy,
            'f1_score': f1
        }

        logging.info(f"Evaluation metrics: {metrics}")
        return metrics

    except Exception as e:
        logging.error(f"Error during evaluation: {e}")
        raise

def save_results(metrics: dict, output_dir: str, filename: str):
    """
    Save evaluation results in JSON and CSV format.

    Args:
        metrics (dict): Evaluation metrics to save.
        output_dir (str): Directory to save the results.
        filename (str): Base filename for the results.
    """
    try:
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # Save as JSON
        json_path = os.path.join(output_dir, f"{filename}.json")
        with open(json_path, 'w') as json_file:
            json.dump(metrics, json_file, indent=4)
        logging.info(f"Results saved to {json_path}")

        # Save as CSV
        csv_path = os.path.join(output_dir, f"{filename}.csv")
        pd.DataFrame([metrics]).to_csv(csv_path, index=False)
        logging.info(f"Results saved to {csv_path}")

    except Exception as e:
        logging.error(f"Error saving results: {e}")
        raise

def plot_metrics(metrics: dict, output_dir: str):
    """
    Generate and save visualizations for evaluation metrics.

    Args:
        metrics (dict): Evaluation metrics to visualize.
        output_dir (str): Directory to save the plots.
    """
    try:
        # Create a bar plot for metrics
        plt.figure(figsize=(8, 5))
        plt.bar(metrics.keys(), metrics.values(), color=['blue', 'orange'])
        plt.ylabel('Score')
        plt.title('Evaluation Metrics')
        plt.ylim(0, 1)
        plt.grid(axis='y')

        # Save the plot
        plot_path = os.path.join(output_dir, 'evaluation_metrics.png')
        plt.savefig(plot_path)
        plt.close()
        logging.info(f"Plot saved to {plot_path}")

    except Exception as e:
        logging.error(f"Error generating plots: {e}")
        raise

# Example usage
if __name__ == "__main__":
    pass
    # Load your model and test_loader here
    # model = ...
    # test_loader = ...
    
    # output_dir = 'evaluation_results'
    # metrics = evaluate(model, test_loader)
    # save_results(metrics, output_dir, 'evaluation_results')
    # plot_metrics(metrics, output_dir)
