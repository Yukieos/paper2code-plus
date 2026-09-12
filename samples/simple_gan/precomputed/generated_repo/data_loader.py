import os
import json
import logging
from typing import Tuple
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DataLoaderModule:
    def __init__(self, batch_size: int):
        self.batch_size = batch_size
        self.train_loader = None
        self.eval_loader = None

    def load_data(self) -> Tuple[DataLoader, DataLoader]:
        """
        Load the training and evaluation datasets, apply preprocessing and augmentation, and return data loaders.

        Args:
            batch_size (int): The number of samples per gradient update.

        Returns:
            Tuple[DataLoader, DataLoader]: Training and evaluation data loaders.
        """
        try:
            # Define transformations
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),  # Normalize to [-1, 1]
            ])

            # Load the MNIST dataset
            train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
            eval_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

            # Create DataLoader instances
            self.train_loader = DataLoader(dataset=train_dataset, batch_size=self.batch_size, shuffle=True)
            self.eval_loader = DataLoader(dataset=eval_dataset, batch_size=self.batch_size, shuffle=False)

            logging.info("Data loaders created successfully.")
            return self.train_loader, self.eval_loader

        except Exception as e:
            logging.error(f"Error loading data: {e}")
            raise

    def save_results(self, results: dict, filename: str):
        """
        Save evaluation results in JSON format.

        Args:
            results (dict): The results to save.
            filename (str): The filename for the output JSON file.
        """
        try:
            with open(filename, 'w') as f:
                json.dump(results, f, indent=4)
            logging.info(f"Results saved to {filename}.")
        except Exception as e:
            logging.error(f"Error saving results: {e}")

    def visualize_results(self, loss_data: dict):
        """
        Generate loss curves from the evaluation metrics.

        Args:
            loss_data (dict): The loss data to visualize.
        """
        try:
            plt.figure(figsize=(10, 5))
            plt.plot(loss_data['generator_loss'], label='Generator Loss')
            plt.plot(loss_data['discriminator_loss'], label='Discriminator Loss')
            plt.title('Loss Curves')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.legend()
            plt.savefig('loss_curves.png')
            plt.close()
            logging.info("Loss curves visualized and saved as 'loss_curves.png'.")
        except Exception as e:
            logging.error(f"Error visualizing results: {e}")

# Example usage
if __name__ == "__main__":
    data_loader_module = DataLoaderModule(batch_size=128)
    train_loader, eval_loader = data_loader_module.load_data()