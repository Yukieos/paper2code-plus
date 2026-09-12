import argparse
import json
import logging
import os
import torch
import torchvision
from data_loaders import get_data_loaders
from models import Generator, Discriminator
from training import train_model
from evaluation import evaluate_model
import pandas as pd
import matplotlib.pyplot as plt

def main(config_path: str, seed: int) -> None:
    """
    Main entry point for the MiniGAN training and evaluation process.
    Accepts a configuration file and a random seed for reproducibility.
    """
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    try:
        # Load configuration
        with open(config_path, 'r') as f:
            config = json.load(f)
        logger.info("Configuration loaded successfully.")

        # Set random seed for reproducibility
        torch.manual_seed(seed)
        logger.info(f"Random seed set to {seed}.")

        # Initialize data loaders
        train_loader, val_loader = get_data_loaders(config['global']['batch_size']['value'])
        logger.info("Data loaders initialized.")

        # Initialize models
        generator = Generator()
        discriminator = Discriminator()
        logger.info("Models initialized.")

        # Train the model
        train_model(generator, discriminator, train_loader, config, logger)

        # Evaluate the model
        metrics = evaluate_model(generator, discriminator, val_loader, logger)

        # Save results
        results_path = 'evaluation_results.json'
        with open(results_path, 'w') as f:
            json.dump(metrics, f)
        logger.info(f"Evaluation results saved to {results_path}.")

        # Generate visualizations
        visualize_results(metrics)

    except Exception as e:
        logger.error(f"An error occurred: {e}")

def visualize_results(metrics):
    """
    Generate visualizations for the evaluation metrics.
    """
    # Example: Plotting loss curves
    plt.figure()
    plt.plot(metrics['generator_loss'], label='Generator Loss')
    plt.plot(metrics['discriminator_loss'], label='Discriminator Loss')
    plt.title('Loss Curves')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('loss_curves.png')
    plt.close()
    logger.info("Loss curves visualized and saved as 'loss_curves.png'.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='MiniGAN Training and Evaluation')
    parser.add_argument('--config_path', type=str, default='config.json', help='Path to configuration file.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility.')
    args = parser.parse_args()

    main(args.config_path, args.seed)