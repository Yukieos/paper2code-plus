import json
import logging
import argparse
import torch

from data_loader import load_data
from model import TinyConvNet, train_model, validate_model
from evaluate import evaluate
from utils import initialize_optimizer, set_seed

def main(config_path: str, seed: int) -> None:
    """Main entry point for the TinyConvNet application. Loads configuration, sets seed, prepares data, trains the model, and evaluates it."""
    
    # Initialize logging
    logging.basicConfig(filename='app.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    # Load configuration from config_path
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
    except FileNotFoundError:
        logger.error(f"Configuration file not found: {config_path}")
        return
    except json.JSONDecodeError:
        logger.error(f"Error decoding JSON from the configuration file: {config_path}")
        return

    # Set random seed for reproducibility
    set_seed(seed)

    # Prepare the dataset
    batch_size = config['global']['batch_size']['value']
    train_loader, test_loader = load_data(batch_size)

    # Initialize the TinyConvNet model
    model = TinyConvNet()

    # Initialize optimizer and criterion
    learning_rate = config['global']['learning_rate']['value']
    optimizer = initialize_optimizer(model, learning_rate)
    criterion = torch.nn.CrossEntropyLoss()

    # Train the model using training data
    num_epochs = config['global']['epochs']['value']
    for epoch in range(num_epochs):
        train_loss = train_model(model, train_loader, None, criterion, optimizer, 1, 'cpu', None)
        logger.info(f"Epoch [{epoch+1}/{num_epochs}], Loss: {train_loss:.4f}")

    # Evaluate the model on test data
    test_accuracy = evaluate(model, test_loader)
    logger.info(f"Test Accuracy: {test_accuracy:.2f}%")

    # Check if the model meets the acceptance criteria
    if test_accuracy < 98.0:
        logger.error("Model did not achieve the required accuracy of 98% on the test dataset.")
    else:
        logger.info("Model achieved the required accuracy on the test dataset.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='TinyConvNet Training and Evaluation')
    parser.add_argument('--config_path', type=str, default='config.json', help='Path to the configuration file.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility.')
    args = parser.parse_args()
    
    main(config_path=args.config_path, seed=args.seed)