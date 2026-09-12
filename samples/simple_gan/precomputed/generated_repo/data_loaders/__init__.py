import logging
from typing import Tuple
import os
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.io import read_image

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CustomDataset(Dataset):
    def __init__(self, data_path: str, transform=None):
        self.data_path = data_path
        self.transform = transform
        self.data = self.load_data()

    def load_data(self) -> np.ndarray:
        if not os.path.exists(self.data_path):
            logger.error(f"Data path {self.data_path} does not exist.")
            raise FileNotFoundError(f"Data path {self.data_path} does not exist.")
        
        try:
            # Assuming the data is in CSV format for this example
            data = pd.read_csv(self.data_path)
            logger.info("Data loaded successfully.")
            return data.values
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> np.ndarray:
        sample = self.data[idx]
        if self.transform:
            sample = self.transform(sample)
        return sample

def get_data_loaders(batch_size: int) -> Tuple[DataLoader, DataLoader]:
    logger.info("Creating data loaders.")
    
    # Define transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),  # Normalization for grayscale images
        transforms.Lambda(lambda x: x.view(-1, 1, 28, 28))  # Reshape to (C, H, W)
    ])

    # Load datasets
    try:
        train_dataset = MNIST(root='./data', train=True, download=True, transform=transform)
        eval_dataset = MNIST(root='./data', train=False, download=True, transform=transform)
        logger.info("Datasets loaded successfully.")
    except Exception as e:
        logger.error(f"Error loading datasets: {e}")
        raise

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)

    logger.info(f"Data loaders created with batch size {batch_size}.")
    return train_loader, eval_loader

def preprocess_data(data: np.ndarray) -> np.ndarray:
    logger.info("Starting data preprocessing.")
    # Example preprocessing: normalization
    data = data / 255.0  # Normalize pixel values to [0, 1]
    logger.info("Data preprocessing completed.")
    return data

# Example usage
if __name__ == "__main__":
    batch_size = 128  # Default batch size
    train_loader, eval_loader = get_data_loaders(batch_size)