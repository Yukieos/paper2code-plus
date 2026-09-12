import argparse
import json
import logging
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import pandas as pd
import matplotlib.pyplot as plt

# Set up logging
logging.basicConfig(level=logging.INFO)

class Generator(nn.Module):
    # Define the generator architecture
    def __init__(self, latent_vector_size):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_vector_size, 128),
            nn.ReLU(),
            nn.Linear(128, 28 * 28),
            nn.Tanh()
        )

    def forward(self, z):
        return self.model(z).view(-1, 1, 28, 28)

class Discriminator(nn.Module):
    # Define the discriminator architecture
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, img):
        return self.model(img)

def load_data(batch_size):
    # Load MNIST dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    eval_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, eval_loader

def save_results(results, filename):
    # Save results to JSON and CSV
    with open(filename + '.json', 'w') as json_file:
        json.dump(results, json_file)
    pd.DataFrame(results).to_csv(filename + '.csv', index=False)

def visualize_losses(generator_losses, discriminator_losses):
    # Generate loss curves
    plt.figure(figsize=(10, 5))
    plt.plot(generator_losses, label='Generator Loss')
    plt.plot(discriminator_losses, label='Discriminator Loss')
    plt.title('Loss Curves')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('loss_curves.png')
    plt.close()

def evaluate_model(generator, discriminator, eval_loader, latent_vector_size):
    # Evaluate the model and return losses
    generator.eval()
    discriminator.eval()
    generator_losses = []
    discriminator_losses = []
    
    with torch.no_grad():
        for imgs, _ in eval_loader:
            z = torch.randn(imgs.size(0), latent_vector_size)
            generated_imgs = generator(z)
            d_loss_real = nn.BCELoss()(discriminator(imgs), torch.ones(imgs.size(0), 1))
            d_loss_fake = nn.BCELoss()(discriminator(generated_imgs), torch.zeros(imgs.size(0), 1))
            d_loss = d_loss_real + d_loss_fake
            
            g_loss = nn.BCELoss()(discriminator(generated_imgs), torch.ones(imgs.size(0), 1))
            
            generator_losses.append(g_loss.item())
            discriminator_losses.append(d_loss.item())
    
    return generator_losses, discriminator_losses

def main(config_path: str, seed: int) -> None:
    # Main entry point for the MiniGAN training and evaluation process.
    logging.info("Loading configuration from %s", config_path)
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
    except FileNotFoundError:
        logging.error("Configuration file not found: %s", config_path)
        return
    except json.JSONDecodeError:
        logging.error("Error decoding JSON from the configuration file: %s", config_path)
        return

    # Validate configuration keys
    required_keys = ['global']
    for key in required_keys:
        if key not in config:
            logging.error("Missing required key in configuration: %s", key)
            return

    # Set random seed for reproducibility
    torch.manual_seed(seed)

    # Initialize data loaders
    batch_size = config['global'].get('batch_size', {}).get('value', 64)
    train_loader, eval_loader = load_data(batch_size)

    # Create model instances
    latent_vector_size = config['global'].get('latent_vector_size', {}).get('value', 100)
    generator = Generator(latent_vector_size)
    discriminator = Discriminator()

    # Define loss function and optimizers
    learning_rate = config['global'].get('learning_rate', {}).get('value', 0.0002)
    optimizer_g = optim.Adam(generator.parameters(), lr=learning_rate)
    optimizer_d = optim.Adam(discriminator.parameters(), lr=learning_rate)

    # Run training loop
    epochs = config['global'].get('epochs', {}).get('value', 10)
    generator_losses = []
    discriminator_losses = []

    for epoch in range(epochs):
        for imgs, _ in train_loader:
            # Train Discriminator
            optimizer_d.zero_grad()
            z = torch.randn(imgs.size(0), latent_vector_size)
            generated_imgs = generator(z)
            d_loss_real = nn.BCELoss()(discriminator(imgs), torch.ones(imgs.size(0), 1))
            d_loss_fake = nn.BCELoss()(discriminator(generated_imgs), torch.zeros(imgs.size(0), 1))
            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()
            optimizer_d.step()

            # Train Generator
            optimizer_g.zero_grad()
            g_loss = nn.BCELoss()(discriminator(generated_imgs), torch.ones(imgs.size(0), 1))
            g_loss.backward()
            optimizer_g.step()

            generator_losses.append(g_loss.item())
            discriminator_losses.append(d_loss.item())

        logging.info(f'Epoch [{epoch+1}/{epochs}], Generator Loss: {g_loss.item()}, Discriminator Loss: {d_loss.item()}')

    # Evaluate model performance
    generator_losses_eval, discriminator_losses_eval = evaluate_model(generator, discriminator, eval_loader, latent_vector_size)
    save_results({
        'generator_losses': generator_losses_eval,
        'discriminator_losses': discriminator_losses_eval
    }, 'evaluation_results')

    # Visualize losses
    visualize_losses(generator_losses_eval, discriminator_losses_eval)

    # Save model weights
    torch.save(generator.state_dict(), 'generator_weights.pth')
    torch.save(discriminator.state_dict(), 'discriminator_weights.pth')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='MiniGAN Training and Evaluation')
    parser.add_argument('--config', type=str, default='config.json', help='Path to the configuration file.')
    parser.add_argument('--seed', type=int, default=42, help='Seed for random number generation.')
    args = parser.parse_args()
    
    main(args.config, args.seed)