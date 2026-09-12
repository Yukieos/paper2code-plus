import os
import logging
from typing import Tuple
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import MNIST
from PIL import Image

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CustomDataset(Dataset):
    def __init__(self, image_dir: str, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.images = os.listdir(image_dir)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path = os.path.join(self.image_dir, self.images[idx])
        image = Image.open(img_path).convert('L')  # Convert to grayscale
        label = int(self.images[idx].split('_')[0])  # Assuming filename format 'label_imageid.png'

        if self.transform:
            image = self.transform(image)

        return image, label

def load_data(batch_size: int, data_augmentation: bool) -> Tuple[DataLoader, DataLoader]:
    try:
        logger.info("Loading dataset...")
        # Define transformations
        transform_list = [transforms.Resize((28, 28)), transforms.ToTensor(), transforms.Lambda(lambda x: x.unsqueeze(0))]
        
        if data_augmentation:
            transform_list.insert(1, transforms.RandomHorizontalFlip())
            transform_list.insert(2, transforms.RandomRotation(10))
        
        transform = transforms.Compose(transform_list)

        # Load training and test datasets
        train_dataset = MNIST(root='./data', train=True, download=True, transform=transform)
        test_dataset = MNIST(root='./data', train=False, download=True, transform=transform)

        # Create DataLoaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        logger.info("Dataset loaded successfully.")
        return train_loader, test_loader

    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        raise

def normalize_data(images: torch.Tensor) -> torch.Tensor:
    """Normalizes the input images to the range [0, 1]."""
    try:
        logger.info("Normalizing data...")
        normalized_images = images / 255.0
        logger.info("Data normalization completed.")
        return normalized_images
    except Exception as e:
        logger.error(f"Error normalizing data: {e}")
        raise

def augment_data(images: torch.Tensor) -> torch.Tensor:
    """Applies data augmentation techniques to the input images."""
    try:
        logger.info("Applying data augmentation...")
        # Placeholder for augmentation logic
        # In practice, augmentations would be applied in the transform pipeline
        augmented_images = images  # No-op for now
        logger.info("Data augmentation completed.")
        return augmented_images
    except Exception as e:
        logger.error(f"Error during data augmentation: {e}")
        raise