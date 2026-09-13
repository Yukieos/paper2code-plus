import logging
from typing import Tuple
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_data(batch_size: int = 64) -> Tuple[DataLoader, DataLoader]:
    """
    Loads the handwritten digit dataset and returns train and test DataLoaders, handling edge cases.

    Args:
        batch_size (int): The batch size for the DataLoader.

    Returns:
        Tuple[DataLoader, DataLoader]: A tuple containing the training and testing DataLoaders.
    """
    logger.info("Starting data loading process...")

    # Define transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # Normalizing to [-1, 1]
    ])

    try:
        # Load the dataset
        train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

        logger.info("Dataset loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        raise

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    logger.info(f"Train DataLoader created with {len(train_loader)} batches.")
    logger.info(f"Test DataLoader created with {len(test_loader)} batches.")

    return train_loader, test_loader

# Example usage
if __name__ == "__main__":
    train_loader, test_loader = load_data(batch_size=64)