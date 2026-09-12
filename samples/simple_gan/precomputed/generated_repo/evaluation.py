import os
import json
import logging
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from scipy.linalg import sqrtm
from typing import List, Tuple
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def calculate_fid(real_images: np.ndarray, generated_images: np.ndarray) -> float:
    """Calculate the Fréchet Inception Distance (FID) between real and generated images."""
    mu_real, sigma_real = real_images.mean(axis=0), np.cov(real_images, rowvar=False)
    mu_gen, sigma_gen = generated_images.mean(axis=0), np.cov(generated_images, rowvar=False)
    
    fid = np.sum((mu_real - mu_gen) ** 2) + np.trace(sigma_real + sigma_gen - 2 * sqrtm(sigma_real @ sigma_gen))
    return fid

def save_results(losses: List[float], fid_score: float, output_path: str):
    """Save evaluation results to JSON and CSV."""
    results = {
        "losses": losses,
        "fid_score": fid_score
    }
    
    # Save as JSON
    with open(os.path.join(output_path, 'evaluation_results.json'), 'w') as json_file:
        json.dump(results, json_file)
    
    # Save as CSV
    with open(os.path.join(output_path, 'evaluation_results.csv'), 'w') as csv_file:
        csv_file.write("losses,fid_score\n")
        for loss in losses:
            csv_file.write(f"{loss},{fid_score}\n")

def plot_loss_curve(losses: List[float], output_path: str):
    """Generate and save a plot of the loss curve."""
    plt.figure()
    plt.plot(losses, label='Generator Loss')
    plt.title('Generator Loss Curve')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(output_path, 'loss_curve.png'))
    plt.close()

def evaluate(generator: nn.Module, eval_loader: DataLoader, real_images: np.ndarray, output_path: str) -> Tuple[List[float], float]:
    """Evaluate the MiniGAN model and return loss curves and FID score.

    Args:
        generator (nn.Module): The generator model.
        eval_loader (DataLoader): Data loader for evaluation data.
        real_images (np.ndarray): Real images for FID calculation.
        output_path (str): Path to save evaluation results.

    Returns:
        Tuple[List[float], float]: List of losses and FID score.
    """
    generator.eval()
    losses = []
    generated_images = []

    with torch.no_grad():
        for i, (images, _) in enumerate(eval_loader):
            noise = torch.randn(images.size(0), 100, 1, 1).to(images.device)  # Assuming latent vector size is 100
            fake_images = generator(noise)
            generated_images.append(fake_images.cpu().numpy())

            # Compute loss (assuming a loss function is defined)
            loss = nn.BCELoss()(fake_images, images)  # Placeholder for actual loss computation
            losses.append(loss.item())

    generated_images = np.concatenate(generated_images, axis=0)
    fid_score = calculate_fid(real_images, generated_images)

    # Save results
    save_results(losses, fid_score, output_path)
    plot_loss_curve(losses, output_path)

    return losses, fid_score

if __name__ == "__main__":
    # Example usage
    # Load your generator model and evaluation data loader here
    # generator = ...
    # eval_loader = ...
    # real_images = ...  # Load real images for FID calculation
    # output_path = "path/to/save/results"

    # try:
    #     evaluate(generator, eval_loader, real_images, output_path)
    # except Exception as e:
    #     logging.error(f"Evaluation failed: {e}")