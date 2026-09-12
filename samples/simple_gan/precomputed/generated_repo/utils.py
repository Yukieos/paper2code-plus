import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import time
import os
from torch.utils.data import DataLoader

def set_seed(seed: int) -> None:
    """Set the random seed for reproducibility across libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def save_model(model: nn.Module, path: str) -> None:
    """Save the trained model to the specified path."""
    torch.save(model.state_dict(), path)

def train_model(generator: nn.Module, discriminator: nn.Module, dataloader: DataLoader, 
                epochs: int, learning_rate: float, log_interval: int, device: str) -> None:
    """Train the MiniGAN model."""
    generator.to(device)
    discriminator.to(device)

    optimizer_g = optim.Adam(generator.parameters(), lr=learning_rate)
    optimizer_d = optim.Adam(discriminator.parameters(), lr=learning_rate)

    criterion = nn.BCELoss()

    for epoch in range(epochs):
        generator.train()
        discriminator.train()
        
        start_time = time.time()
        for batch_idx, (real_images, _) in enumerate(dataloader):
            real_images = real_images.to(device)
            batch_size = real_images.size(0)

            # Train Discriminator
            optimizer_d.zero_grad()
            real_labels = torch.ones(batch_size, 1).to(device)
            fake_labels = torch.zeros(batch_size, 1).to(device)

            outputs = discriminator(real_images)
            d_loss_real = criterion(outputs, real_labels)

            noise = torch.randn(batch_size, 100).to(device)  # Assuming latent vector size is 100
            fake_images = generator(noise)
            outputs = discriminator(fake_images.detach())
            d_loss_fake = criterion(outputs, fake_labels)

            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()
            optimizer_d.step()

            # Train Generator
            optimizer_g.zero_grad()
            outputs = discriminator(fake_images)
            g_loss = criterion(outputs, real_labels)
            g_loss.backward()
            optimizer_g.step()

            if batch_idx % log_interval == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Step [{batch_idx}/{len(dataloader)}], '
                      f'D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}')

        end_time = time.time()
        print(f'Epoch [{epoch+1}/{epochs}] completed in {end_time - start_time:.2f} seconds.')

        # Save model checkpoints
        if (epoch + 1) % 10 == 0:  # Save every 10 epochs
            save_model(generator, f'generator_epoch_{epoch+1}.pth')
            save_model(discriminator, f'discriminator_epoch_{epoch+1}.pth')

    # Final model save
    save_model(generator, 'generator_final.pth')
    save_model(discriminator, 'discriminator_final.pth')