import os
import json
import logging
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(level=logging.INFO)

class DataLoaderUtils:
    def __init__(self, dataset_path: str):
        self.dataset_path = dataset_path

    def load_dataset(self, split: str) -> Dataset:
        """
        Load and preprocess the dataset for training or evaluation.

        Args:
            split (str): Specify whether to load the training or evaluation dataset.

        Returns:
            Dataset: The loaded and preprocessed dataset.
        """
        try:
            # Load the dataset from the specified source
            dataset_file = os.path.join(self.dataset_path, f"{split}.json")
            with open(dataset_file, 'r') as f:
                data = json.load(f)

            # Preprocess the dataset (tokenization, padding, etc.)
            dataset = self.preprocess_data(data)

            logging.info(f"Loaded {split} dataset with {len(dataset)} samples.")
            return dataset

        except FileNotFoundError:
            logging.error(f"Dataset file not found: {dataset_file}")
            raise
        except Exception as e:
            logging.error(f"An error occurred while loading the dataset: {str(e)}")
            raise

    def preprocess_data(self, data):
        # Implement your preprocessing logic here
        # For example, tokenization and padding
        # This is a placeholder implementation
        return data

    def evaluate_model(self, model, dataloader: DataLoader, metrics: list) -> dict:
        """
        Evaluate the model on the provided dataloader using specified metrics.

        Args:
            model: The model to evaluate.
            dataloader (DataLoader): The dataloader for the evaluation dataset.
            metrics (list): List of metrics to compute.

        Returns:
            dict: A dictionary containing the computed metrics.
        """
        model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in dataloader:
                inputs, labels = batch
                outputs = model(inputs)
                preds = torch.argmax(outputs, dim=1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        results = {}
        if 'accuracy' in metrics:
            results['accuracy'] = accuracy_score(all_labels, all_preds)
        if 'f1' in metrics:
            results['f1'] = f1_score(all_labels, all_preds, average='weighted')

        logging.info(f"Evaluation results: {results}")
        return results

    def save_results(self, results: dict, output_path: str):
        """
        Save evaluation results in structured format (JSON, CSV).

        Args:
            results (dict): The evaluation results to save.
            output_path (str): The path to save the results.
        """
        try:
            with open(output_path, 'w') as f:
                json.dump(results, f)
            logging.info(f"Results saved to {output_path}")
        except Exception as e:
            logging.error(f"An error occurred while saving results: {str(e)}")
            raise

    def generate_visualization(self, results: dict, output_path: str):
        """
        Generate visualizations based on the evaluation results.

        Args:
            results (dict): The evaluation results.
            output_path (str): The path to save the visualization.
        """
        try:
            plt.figure()
            plt.bar(results.keys(), results.values())
            plt.xlabel('Metrics')
            plt.ylabel('Scores')
            plt.title('Evaluation Metrics')
            plt.savefig(output_path)
            plt.close()
            logging.info(f"Visualization saved to {output_path}")
        except Exception as e:
            logging.error(f"An error occurred while generating visualization: {str(e)}")
            raise

# Example usage:
# data_loader_utils = DataLoaderUtils('/path/to/dataset')
# dataset = data_loader_utils.load_dataset('validation')
# results = data_loader_utils.evaluate_model(model, DataLoader(dataset), ['accuracy', 'f1'])
# data_loader_utils.save_results(results, 'results.json')
# data_loader_utils.generate_visualization(results, 'results.png')