import json
import random
import torch
import argparse
from data_loader import SentimentDataset
from model import MiniEncoder
from training import train
from evaluation import evaluate  # Assuming there's an evaluation module
from utils import validate_input, prepare_data

def main(config_path: str, seed: int) -> dict:
    """Main entry point for the MiniEncoder training and evaluation process.

    Args:
        config_path (str): Path to the configuration file.
        seed (int): Random seed for reproducibility.

    Returns:
        dict: Evaluation results after training the model.
    """
    # Load configuration from the specified JSON file.
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    # Set the random seed for reproducibility.
    random.seed(seed)
    torch.manual_seed(seed)

    # Prepare dataset
    dataset = SentimentDataset(file_path=config['data_loader']['file_path'])
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=config['global']['batch_size']['value'], shuffle=True)

    # Initialize model
    model = MiniEncoder(
        vocab_size=config['global']['vocab_size']['value'],
        embedding_dim=config['global']['embedding_dim']['value'],
        num_heads=config['global']['num_heads']['value'],
        head_dim=config['global']['head_dim']['value'],
        dropout_rate=config['global']['dropout_rate']['value']
    )

    # Train model
    optimizer = torch.optim.Adam(model.parameters(), lr=config['global']['learning_rate']['value'])
    criterion = torch.nn.CrossEntropyLoss()
    train(model, train_loader, optimizer, criterion, config['global']['epochs']['value'])

    # Evaluate model
    evaluation_results = evaluate(model, dataset)  # Assuming evaluate function returns a dict of results
    return evaluation_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train and evaluate the MiniEncoder model.')
    parser.add_argument('--config_path', type=str, default='config.json', help='Path to the configuration file.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility.')
    args = parser.parse_args()

    results = main(config_path=args.config_path, seed=args.seed)
    print("Evaluation Results:", results)