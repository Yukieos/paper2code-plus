import os
import json
import logging
import pandas as pd
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(level=logging.INFO)

def load_dataset(dataset_path):
    try:
        data = pd.read_csv(dataset_path)
        logging.info(f"Loaded dataset from {dataset_path} with {len(data)} samples.")
        return data
    except Exception as e:
        logging.error(f"Error loading dataset: {e}")
        raise

def load_model(checkpoint_path):
    try:
        model = DistilBertForSequenceClassification.from_pretrained(checkpoint_path)
        tokenizer = DistilBertTokenizer.from_pretrained(checkpoint_path)
        logging.info(f"Loaded model from {checkpoint_path}.")
        return model, tokenizer
    except Exception as e:
        logging.error(f"Error loading model: {e}")
        raise

def evaluate_model(model, tokenizer, data):
    model.eval()
    predictions, true_labels = [], []
    
    for index, row in data.iterrows():
        inputs = tokenizer(row['text'], return_tensors='pt', padding=True, truncation=True)
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            predicted_class = torch.argmax(logits, dim=1).item()
            predictions.append(predicted_class)
            true_labels.append(row['label'])
    
    metrics = {
        "accuracy": accuracy_score(true_labels, predictions),
        "f1_score": f1_score(true_labels, predictions, average='weighted'),
        "precision": precision_score(true_labels, predictions, average='weighted'),
        "recall": recall_score(true_labels, predictions, average='weighted')
    }
    
    logging.info(f"Evaluation metrics: {metrics}")
    return metrics

def save_results(metrics, output_path):
    try:
        with open(output_path, 'w') as f:
            json.dump(metrics, f)
        logging.info(f"Saved evaluation results to {output_path}.")
    except Exception as e:
        logging.error(f"Error saving results: {e}")
        raise

def plot_metrics(metrics, output_path):
    try:
        plt.figure(figsize=(10, 5))
        plt.bar(metrics.keys(), metrics.values())
        plt.title('Evaluation Metrics')
        plt.xlabel('Metrics')
        plt.ylabel('Scores')
        plt.savefig(output_path)
        plt.close()
        logging.info(f"Saved metrics plot to {output_path}.")
    except Exception as e:
        logging.error(f"Error plotting metrics: {e}")
        raise

def evaluate(config: dict) -> dict:
    """
    Evaluate the DistilBERT model based on the provided configuration.
    
    Args:
        config (dict): Configuration settings for evaluation.
    
    Returns:
        dict: Evaluation metrics computed from the dataset.
    """
    try:
        # Load dataset
        dataset_path = config['dataset']['path']
        data = load_dataset(dataset_path)

        # Load the trained model
        checkpoint_path = config['model']['checkpoint']
        model, tokenizer = load_model(checkpoint_path)

        # Evaluate the model
        metrics = evaluate_model(model, tokenizer, data)

        # Save results
        results_path = config['results']['output_path']
        save_results(metrics, results_path)

        # Generate plots if specified
        if config.get('visualization', {}).get('enabled', False):
            plot_path = config['visualization']['plot_path']
            plot_metrics(metrics, plot_path)

        return metrics

    except Exception as e:
        logging.error(f"Evaluation failed: {e}")
        raise

if __name__ == "__main__":
    # Example configuration
    config = {
        "dataset": {
            "path": "data/validation.csv"
        },
        "model": {
            "checkpoint": "path/to/checkpoint"
        },
        "results": {
            "output_path": "results/evaluation_metrics.json"
        },
        "visualization": {
            "enabled": True,
            "plot_path": "results/evaluation_plot.png"
        }
    }
    
    evaluate(config)