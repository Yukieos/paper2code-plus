import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import logging
import time
import os
import numpy as np

# Set up logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def train(generator: nn.Module, discriminator: nn.Module, train_loader: DataLoader, epochs: int, learning_rate: float) -> None:
    """
    Trains the MiniGAN model using the specified generator and discriminator, with error handling for convergence issues.

    Args:
        generator (nn.Module): The generator model.
        discriminator (nn.Module): The discriminator model.
        train_loader (DataLoader): DataLoader for the training dataset.
        epochs (int): Number of epochs to train the model.
        learning_rate (float): Learning rate for the optimizers.
    """
    
    # Validate learning rate and epochs
    if learning_rate not in [0.0001, 0.0002, 0.001]:
        raise ValueError(f"Learning rate must be one of [0.0001, 0.0002, 0.001], got {learning_rate}.")
    if epochs not in [10, 20, 50]:
        raise ValueError(f"Epochs must be one of [10, 20, 50], got {epochs}.")

    # Set the model to training mode
    generator.train()
    discriminator.train()

    # Initialize optimizers
    optimizer_g = optim.Adam(generator.parameters(), lr=learning_rate)
    optimizer_d = optim.Adam(discriminator.parameters(), lr=learning_rate)

    # Initialize learning rate scheduler
    scheduler_g = optim.lr_scheduler.StepLR(optimizer_g, step_size=10, gamma=0.1)
    scheduler_d = optim.lr_scheduler.StepLR(optimizer_d, step_size=10, gamma=0.1)

    # Initialize loss function
    criterion = nn.BCELoss()

    # Prepare for logging
    loss_values = {'generator': [], 'discriminator': []}
    start_time = time.time()

    for epoch in range(epochs):
        epoch_start_time = time.time()
        for i, (real_images, _) in enumerate(train_loader):
            real_images = real_images.to(device)

            # Create labels
            batch_size = real_images.size(0)
            real_labels = torch.ones(batch_size, 1).to(device)
            fake_labels = torch.zeros(batch_size, 1).to(device)

            # Train Discriminator
            optimizer_d.zero_grad()

            # Compute loss with real images
            outputs = discriminator(real_images)
            d_loss_real = criterion(outputs, real_labels)

            # Compute loss with fake images
            noise = torch.randn(batch_size, 100).to(device)  # Assuming latent vector size is 100
            fake_images = generator(noise)
            outputs = discriminator(fake_images.detach())
            d_loss_fake = criterion(outputs, fake_labels)

            # Backpropagation and optimization
            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()
            optimizer_d.step()

            # Check for NaN loss
            if np.isnan(d_loss.item()):
                logging.error("Discriminator loss is NaN. Stopping training.")
                return

            # Train Generator
            optimizer_g.zero_grad()

            # Compute loss for fake images
            outputs = discriminator(fake_images)
            g_loss = criterion(outputs, real_labels)

            # Backpropagation and optimization
            g_loss.backward()
            optimizer_g.step()

            # Check for NaN loss
            if np.isnan(g_loss.item()):
                logging.error("Generator loss is NaN. Stopping training.")
                return

            # Log losses
            loss_values['generator'].append(g_loss.item())
            loss_values['discriminator'].append(d_loss.item())

            if (i + 1) % 100 == 0:  # Log every 100 batches
                logging.info(f'Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(train_loader)}], '
                             f'Generator Loss: {g_loss.item():.4f}, Discriminator Loss: {d_loss.item():.4f}')

        # Step the learning rate scheduler
        scheduler_g.step()
        scheduler_d.step()

        # Save model checkpoints
        checkpoint_dir = 'checkpoints'
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        torch.save(generator.state_dict(), os.path.join(checkpoint_dir, f'generator_epoch_{epoch+1}.pth'))
        torch.save(discriminator.state_dict(), os.path.join(checkpoint_dir, f'discriminator_epoch_{epoch+1}.pth'))

        epoch_time = time.time() - epoch_start_time
        logging.info(f'Epoch [{epoch+1}/{epochs}] completed in {epoch_time:.2f} seconds.')

    total_time = time.time() - start_time
    logging.info(f'Training completed in {total_time:.2f} seconds.')