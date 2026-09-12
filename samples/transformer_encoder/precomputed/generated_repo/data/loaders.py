import os
import logging
import pandas as pd
import json
from typing import List, Tuple, Union
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_dataset(data_path: str) -> Tuple[List[str], List[int]]:
    """
    Load the dataset from the specified path and return texts and labels.

    Args:
        data_path (str): Path to the dataset file.

    Returns:
        Tuple[List[str], List[int]]: A tuple containing a list of texts and a list of corresponding labels.
    """
    if not os.path.exists(data_path):
        logging.error(f"Data path {data_path} does not exist.")
        raise FileNotFoundError(f"Data path {data_path} does not exist.")

    try:
        if data_path.endswith('.csv'):
            logging.info("Loading CSV file.")
            data = pd.read_csv(data_path)
        elif data_path.endswith('.json'):
            logging.info("Loading JSON file.")
            with open(data_path, 'r') as f:
                data = json.load(f)
        elif data_path.endswith('.h5'):
            logging.info("Loading HDF5 file.")
            data = pd.read_hdf(data_path)
        else:
            logging.error("Unsupported file format.")
            raise ValueError("Unsupported file format. Please use CSV, JSON, or HDF5.")

        # Validate dataset schema
        if 'text' not in data.columns or 'label' not in data.columns:
            logging.error("Dataset must contain 'text' and 'label' columns.")
            raise ValueError("Dataset must contain 'text' and 'label' columns.")

        texts = data['text'].tolist()
        labels = data['label'].tolist()

        # Tokenize the text data
        tokenized_texts = tokenize_data(texts)

        return tokenized_texts, labels

    except Exception as e:
        logging.exception("An error occurred while loading the dataset.")
        raise e

def tokenize_data(raw_texts: List[str]) -> List[str]:
    """
    Tokenize raw text data into a list of tokenized strings.

    Args:
        raw_texts (List[str]): List of raw text strings to be tokenized.

    Returns:
        List[str]: A list of tokenized strings.
    """
    tokenized_texts = []
    for text in tqdm(raw_texts, desc="Tokenizing texts"):
        try:
            # Lowercase the text
            text = text.lower()
            # Split the text into tokens based on whitespace and punctuation
            tokens = text.split()  # Simple whitespace tokenization
            tokenized_texts.append(' '.join(tokens))
        except Exception as e:
            logging.error(f"Error tokenizing text: {text}")
            logging.exception(e)

    return tokenized_texts

# Example usage
if __name__ == "__main__":
    # Load dataset (example path)
    try:
        texts, labels = load_dataset('data/dataset.csv')
        logging.info(f"Loaded {len(texts)} texts and {len(labels)} labels.")
    except Exception as e:
        logging.error("Failed to load dataset.")