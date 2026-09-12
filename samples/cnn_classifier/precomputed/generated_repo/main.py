import json
import torch
import logging
from data.loaders import load_data, normalize_images
from evaluation.evaluate import evaluate_model
from models.tiny_conv_net import TinyConvNet, train_model, validate_model
from utils.report import report_results

def setup_logging():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    return logging.getLogger(__name__)

def main(config_path: str = "config.json", seed: int = 42) -> None:
    """Main entry point for the TinyConvNet application. Loads configuration, sets seed, and orchestrates the training and evaluation process."""
    
    logger = setup_logging()

    # Load configuration from config_path
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
    except FileNotFoundError:
        logger.error(f"Configuration file not found: {config_path}")
        return
    except json.JSONDecodeError:
        logger.error(f"Configuration file is malformed: {config_path}")
        return

    # Set random seed for reproducibility
    torch.manual_seed(seed)

    # Load dataset and create data loaders
    batch_size = config['global']['batch_size']['value']
    train_loader, test_loader = load_data(batch_size)

    if len(train_loader) == 0 or len(test_loader) == 0:
        logger.error("Loaded dataset is empty. Please check the data source.")
        raise ValueError("Loaded dataset is empty.")

    # Initialize TinyConvNet model
    try:
        model = TinyConvNet()
    except RuntimeError as e:
        logger.error(f"Error initializing the model: {e}")
        return

    optimizer = torch.optim.Adam(model.parameters(), lr=config['global']['learning_rate']['value'])
    criterion = torch.nn.CrossEntropyLoss()

    # Train the model
    num_epochs = config['global']['epochs']['value']
    logger.info("Starting training...")
    train_model(model, train_loader, optimizer, criterion, num_epochs, device='cpu', save_path=None, log_interval=10)

    # Evaluate the model
    logger.info("Evaluating the model...")
    accuracy, confusion_matrix, precision, recall, f1_score = evaluate_model(model, test_loader)

    # Report the results
    logger.info("Reporting results...")
    report_results(accuracy, confusion_matrix, precision, recall, f1_score)

if __name__ == "__main__":
    main()