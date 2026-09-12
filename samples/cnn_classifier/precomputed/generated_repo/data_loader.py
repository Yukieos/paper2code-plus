import os
import logging
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from typing import Any, Callable, List, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HandwrittenDigitDataset(Dataset):
    """
    Dataset class for loading and preprocessing the handwritten digit dataset.
    """

    def __init__(self, data_dir: str, transform: Callable = None) -> None:
        """
        Initialize the dataset.

        Args:
            data_dir (str): Directory containing the dataset.
            transform (callable, optional): Transform to apply to the images.
        """
        self.data_dir = data_dir
        self.transform = transform
        self.images: List[np.ndarray] = []
        self.labels: List[int] = []

        self.load_data()

    def load_data(self) -> None:
        """
        Load raw images and labels from the dataset directory.
        """
        logger.info("Loading data from %s", self.data_dir)

        try:
            # Assuming images are in a folder and labels are in a CSV file
            label_file = os.path.join(self.data_dir, 'labels.csv')
            labels_df = pd.read_csv(label_file)

            for index, row in labels_df.iterrows():
                image_path = os.path.join(self.data_dir, row['filename'])
                if os.path.exists(image_path):
                    image = Image.open(image_path).convert('L')  # Convert to grayscale
                    self.images.append(np.array(image))
                    self.labels.append(row['label'])
                else:
                    logger.warning("Image %s not found, skipping.", image_path)

            logger.info("Loaded %d images and labels.", len(self.images))

        except Exception as e:
            logger.error("Error loading data: %s", e)
            raise

    def __len__(self) -> int:
        """
        Return the total number of samples.

        Returns:
            int: Total number of samples.
        """
        return len(self.images)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, int]:
        """
        Retrieve an image and its corresponding label.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            Tuple[np.ndarray, int]: Tuple containing the image and its label.
        """
        if idx < 0 or idx >= len(self.images):
            logger.error("Index %d out of bounds", idx)
            raise IndexError("Index out of bounds")

        image = self.images[idx]
        label = self.labels[idx]

        # Normalize images to [0, 1] range
        image = image.astype(np.float32) / 255.0

        if self.transform:
            image = self.transform(image)

        return image, label

# Example of a preprocessing pipeline
def get_transform() -> Callable:
    """
    Get the preprocessing transformations.

    Returns:
        Callable: Transformations to apply to the images.
    """
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),  # Normalize to [-1, 1]
    ])