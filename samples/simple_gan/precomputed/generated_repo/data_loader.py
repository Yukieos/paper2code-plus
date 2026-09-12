import os
import logging
from typing import Tuple
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np  # Added import for numpy

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HandwrittenDigitDataset:
    def __init__(self, batch_size: int):
        self.batch_size = batch_size
        self.train_loader = None
        self.test_loader = None
        self.data_dir = './data'

    def load_data(self) -> Tuple[DataLoader, DataLoader]:
        """
        Loads the handwritten digit dataset and returns data loaders for training and testing, handling edge cases.
        """
        try:
            # Check if the dataset already exists
            if not os.path.exists(os.path.join(self.data_dir, 'MNIST')):  # Check for dataset directory
                logger.info("Downloading the MNIST dataset...")
                train_dataset = datasets.MNIST(root=self.data_dir, train=True, download=True, transform=self.get_transforms())
                test_dataset = datasets.MNIST(root=self.data_dir, train=False, download=True, transform=self.get_transforms())
            else:
                logger.info("MNIST dataset already exists. Loading from disk...")
                train_dataset = datasets.MNIST(root=self.data_dir, train=True, download=False, transform=self.get_transforms())
                test_dataset = datasets.MNIST(root=self.data_dir, train=False, download=False, transform=self.get_transforms())

            # Validate data
            self.validate_data(train_dataset)
            self.validate_data(test_dataset)

            # Create data loaders
            self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
            self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

            logger.info("Data loading complete. Train and test loaders are ready.")
            return self.train_loader, self.test_loader

        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            raise

    def get_transforms(self) -> transforms.Compose:
        """
        Returns the transformation pipeline for the dataset.
        """
        try:
            return transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))  # Normalize to [-1, 1]
            ])
        except Exception as e:
            logger.error(f"Error in transformations: {e}")
            raise

    def validate_data(self, dataset):
        """
        Validates the dataset for missing or corrupted images.
        """
        if len(dataset) == 0:  # Check if the dataset is empty
            logger.warning("The dataset is empty.")
            return
        
        for i, (image, label) in enumerate(dataset):
            if image is None or label is None:
                logger.warning(f"Missing data at index {i}.")
            if not isinstance(image, torch.Tensor):
                logger.warning(f"Corrupted image at index {i}. Expected tensor, got {type(image)}.")

# Example usage
if __name__ == "__main__":
    dataset_loader = HandwrittenDigitDataset(batch_size=128)
    train_loader, test_loader = dataset_loader.load_data()