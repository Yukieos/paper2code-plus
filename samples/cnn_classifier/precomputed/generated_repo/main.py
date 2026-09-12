import json
import logging
import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from model import TinyConvNet  # Assuming model.py contains the TinyConvNet definition
from training import train_model  # Assuming training.py contains the training logic
from evaluation import evaluate_model  # Assuming evaluation.py contains the evaluation logic

def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info("Starting the evaluation pipeline.")

def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)

def prepare_dataset(data_path: str, batch_size: int) -> DataLoader:
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = datasets.MNIST(root=data_path, train=False, transform=transform, download=True)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return data_loader

def initialize_model() -> TinyConvNet:
    model = TinyConvNet()
    return model

def report_results(results: dict, output_path: str) -> None:
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=4)
    logging.info(f"Results saved to {output_path}")

def handle_data_errors(data: Any) -> None:
    if data.isnull().values.any():
        logging.error("Data contains missing values.")
        raise ValueError("Data contains missing values.")
    logging.info("Data integrity check passed.")

def main(config_path: str = "config.json", seed: int = 42) -> None:
    setup_logging()
    
    try:
        config = load_config(config_path)
        set_seed(seed)

        data_loader = prepare_dataset(config['data_path'], config['global']['batch_size']['value'])
        model = initialize_model()

        checkpoint_path = config['checkpoint_path']
        if os.path.exists(checkpoint_path):
            model.load_state_dict(torch.load(checkpoint_path))
            logging.info(f"Model loaded from {checkpoint_path}")
        else:
            logging.error(f"Checkpoint not found at {checkpoint_path}")
            sys.exit(1)

        results = evaluate_model(model, data_loader)
        report_results(results, config['output_path'])

    except Exception as e:
        logging.error(f"An error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TinyConvNet Evaluation Pipeline")
    parser.add_argument('--config_path', type=str, default="config.json", help="Path to the configuration file.")
    parser.add_argument('--seed', type=int, default=42, help="Random seed for reproducibility.")
    args = parser.parse_args()
    
    main(args.config_path, args.seed)