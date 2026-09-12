import logging
import json
import csv
import os
from typing import List, Tuple
import torch
import torch.nn as nn
from metrics import calculate_fid, calculate_loss  # Assuming these functions are defined in metrics module

logger = logging.getLogger(__name__)

def evaluate(generator: nn.Module, eval_loader: torch.utils.data.DataLoader) -> Tuple[List[float], float]:
    """
    Evaluates the MiniGAN model and returns loss curves and FID score, with error handling for evaluation issues.

    Args:
        generator (nn.Module): The generator model to evaluate.
        eval_loader (DataLoader): DataLoader for the evaluation dataset.

    Returns:
        Tuple[List[float], float]: A tuple containing a list of losses and the FID score.
    """
    try:
        generator.eval()  # Set the model to evaluation mode
        losses = []
        fid_score = 0.0

        with torch.no_grad():  # Disable gradient calculation
            for batch in eval_loader:
                # Assuming batch contains images and labels
                images, _ = batch
                generated_images = generator(images)

                # Calculate loss and FID score
                loss = calculate_loss(generated_images, images)  # Placeholder for actual loss calculation
                losses.append(loss.item())
                fid_score = calculate_fid(generated_images, images)  # Placeholder for actual FID calculation

        logger.info(f'Evaluation completed. Average Loss: {sum(losses)/len(losses)}, FID Score: {fid_score}')
        return losses, fid_score

    except Exception as e:
        logger.error(f'Error during evaluation: {str(e)}')
        raise

def save_results(losses: List[float], fid_score: float, output_dir: str) -> None:
    """
    Saves the evaluation results in structured format (JSON, CSV).

    Args:
        losses (List[float]): List of losses for each batch.
        fid_score (float): FID score for the generated images.
        output_dir (str): Directory to save the results.
    """
    try:
        # Save losses to JSON
        json_path = os.path.join(output_dir, 'evaluation_results.json')
        with open(json_path, 'w') as json_file:
            json.dump({'losses': losses, 'fid_score': fid_score}, json_file)

        # Save losses to CSV
        csv_path = os.path.join(output_dir, 'evaluation_results.csv')
        with open(csv_path, 'w', newline='') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(['Batch', 'Loss'])
            for i, loss in enumerate(losses):
                writer.writerow([i, loss])

        logger.info(f'Results saved to {json_path} and {csv_path}')

    except Exception as e:
        logger.error(f'Error saving results: {str(e)}')
        raise

def generate_visualizations(losses: List[float], fid_score: float, output_dir: str) -> None:
    """
    Generates visualizations for the evaluation metrics if mentioned in the paper.

    Args:
        losses (List[float]): List of losses for each batch.
        fid_score (float): FID score for the generated images.
        output_dir (str): Directory to save the visualizations.
    """
    try:
        import matplotlib.pyplot as plt

        # Plot loss curves
        plt.figure()
        plt.plot(losses, label='Loss')
        plt.title('Loss Curve')
        plt.xlabel('Batch')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(os.path.join(output_dir, 'loss_curve.png'))
        plt.close()

        logger.info(f'Visualizations saved to {output_dir}')

    except Exception as e:
        logger.error(f'Error generating visualizations: {str(e)}')
        raise

def main(generator: nn.Module, eval_loader: torch.utils.data.DataLoader, output_dir: str) -> None:
    """
    Main evaluation function to run the evaluation process.

    Args:
        generator (nn.Module): The generator model to evaluate.
        eval_loader (DataLoader): DataLoader for the evaluation dataset.
        output_dir (str): Directory to save the results and visualizations.
    """
    losses, fid_score = evaluate(generator, eval_loader)
    save_results(losses, fid_score, output_dir)
    generate_visualizations(losses, fid_score, output_dir)