import os
import json
import logging
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score
from model import MiniEncoder  # Assuming the model is defined in model.py
from dataset import load_test_dataset  # Assuming a function to load the test dataset

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_model(checkpoint_path: str) -> MiniEncoder:
    """Load the trained MiniEncoder model from a checkpoint."""
    if not os.path.exists(checkpoint_path):
        logging.error(f"Checkpoint path {checkpoint_path} does not exist.")
        raise FileNotFoundError(f"Checkpoint path {checkpoint_path} does not exist.")
    
    model = MiniEncoder()  # Initialize the model
    model.load_state_dict(torch.load(checkpoint_path))
    model.eval()  # Set the model to evaluation mode
    logging.info("Model loaded successfully from checkpoint.")
    return model

def evaluate_model(model: MiniEncoder, test_loader: DataLoader) -> dict:
    """Evaluate the trained MiniEncoder model on the test dataset."""
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    
    logging.info(f"Evaluation results - Accuracy: {accuracy}, F1 Score: {f1}")
    return {"accuracy": accuracy, "f1_score": f1}

def save_results(results: dict, output_path: str):
    """Save evaluation results in JSON and CSV formats."""
    with open(output_path + '.json', 'w') as json_file:
        json.dump(results, json_file)
    logging.info(f"Results saved to {output_path}.json")
    
    df = pd.DataFrame([results])
    df.to_csv(output_path + '.csv', index=False)
    logging.info(f"Results saved to {output_path}.csv")

def main(config_path: str, checkpoint_path: str, output_path: str):
    """Main evaluation function."""
    # Load configuration
    with open(config_path, 'r') as config_file:
        config = json.load(config_file)
    
    # Load test dataset
    test_loader = load_test_dataset(batch_size=config['global']['batch_size']['value'])
    
    # Load trained model
    model = load_model(checkpoint_path)
    
    # Evaluate model
    results = evaluate_model(model, test_loader)
    
    # Output evaluation metrics
    save_results(results, output_path)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate the MiniEncoder model.")
    parser.add_argument('--config', type=str, required=True, help='Path to the configuration file.')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to the model checkpoint.')
    parser.add_argument('--output', type=str, required=True, help='Base path for output results.')
    
    args = parser.parse_args()
    
    try:
        main(args.config, args.checkpoint, args.output)
    except Exception as e:
        logging.error(f"An error occurred during evaluation: {e}")