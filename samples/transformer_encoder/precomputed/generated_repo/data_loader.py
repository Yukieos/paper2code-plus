import pandas as pd
import logging
from typing import List, Tuple
from torch.utils.data import Dataset

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SentimentDataset(Dataset):
    def __init__(self, file_path: str):
        """
        Initialize the dataset by loading the data from the specified file path.

        Args:
            file_path (str): Path to the dataset file.
        """
        self.texts, self.labels = self.load_data(file_path)

    def __len__(self) -> int:
        """Return the total number of samples in the dataset."""
        return len(self.texts)

    def __getitem__(self, idx: int) -> Tuple[str, int]:
        """
        Retrieve a sample from the dataset.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            Tuple[str, int]: A tuple containing the text and its corresponding label.
        """
        return self.texts[idx], self.labels[idx]

    def load_data(self, file_path: str) -> Tuple[List[str], List[int]]:
        """
        Load and preprocess the sentiment classification dataset.

        Args:
            file_path (str): Path to the dataset file.

        Returns:
            Tuple[List[str], List[int]]: A tuple containing the list of texts and their corresponding labels.
        """
        logger.info(f"Loading data from {file_path}")
        try:
            # Read the dataset file
            data = pd.read_csv(file_path)
            logger.info("Data loaded successfully")

            # Validate the schema
            if 'text' not in data.columns or 'label' not in data.columns:
                raise ValueError("Dataset must contain 'text' and 'label' columns.")

            # Extract texts and labels
            texts = data['text'].tolist()
            labels = data['label'].tolist()

            # Check for null values
            if any(pd.isnull(text) for text in texts) or any(pd.isnull(label) for label in labels):
                raise ValueError("Dataset contains null values.")

            logger.info("Data validation passed")
            return texts, labels

        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise

# Example usage (uncomment to use):
# dataset = SentimentDataset(file_path='data/dataset.csv')