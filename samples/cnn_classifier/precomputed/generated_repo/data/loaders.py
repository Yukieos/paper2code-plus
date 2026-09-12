import logging
from typing import Tuple
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import datasets

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_data(batch_size: int = 64) -> Tuple[DataLoader, DataLoader]:
    """
    Loads the handwritten digit dataset and returns data loaders for training and testing,
    including error handling for empty datasets.

    Args:
        batch_size (int): The batch size for the data loaders.

    Returns:
        Tuple[DataLoader, DataLoader]: Training and testing data loaders.
    """
    try:
        # Download the dataset
        logger.info("Downloading the MNIST dataset...")
        dataset = datasets.MNIST(root='./data', train=True, download=True)

        # Check for empty dataset
        if len(dataset) == 0:
            logger.error("The dataset is empty.")
            raise ValueError("The dataset is empty.")

        # Apply normalization and transformations
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: normalize_images(x.numpy()))
        ])
        
        # Create DataLoader instances for training and testing
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        logger.info(f"Created training DataLoader with batch size {batch_size}.")

        test_dataset = datasets.MNIST(root='./data', train=False, download=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        logger.info(f"Created testing DataLoader with batch size {batch_size}.")

        return train_loader, test_loader

    except Exception as e:
        logger.exception("An error occurred while loading the dataset.")
        raise e

def normalize_images(images: np.ndarray) -> np.ndarray:
    """
    Normalizes the pixel values of the images to the range [0, 1].

    Args:
        images (np.ndarray): Raw grayscale images.

    Returns:
        np.ndarray: Normalized images.
    """
    # Normalize pixel values to [0, 1]
    normalized_images = images / 255.0
    return normalized_images