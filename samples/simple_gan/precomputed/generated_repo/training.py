import torch
import torch.nn as nn
import torch.optim as optim
import time
import os
import logging
from torch.utils.data import DataLoader

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def train(generator: nn.Module, discriminator: nn.Module, train_loader: DataLoader, epochs: int, learning_rate: float) -> None:
    """
    Train the MiniGAN model using the specified training data.

    Args:
        generator (nn.Module): The generator model.
        discriminator (nn.Module): The discriminator model.
        train_loader (DataLoader): Data loader for training data.
        epochs (int): Number of training epochs.
        learning_rate (float): Learning rate for the optimizer.
    """
    # Initialize lists to store losses for visualization
    g_losses = []
    d_losses = []

    # Move models to the specified device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    generator.to(device)
    discriminator.to(device)

    # Initialize optimizers
    optimizer_g = optim.Adam(generator.parameters(), lr=learning_rate)
    optimizer_d = optim.Adam(discriminator.parameters(), lr=learning_rate)

    # Loss function
    criterion = nn.BCELoss()

    # Training loop
    for epoch in range(epochs):
        generator.train()
        discriminator.train()
        
        epoch_start_time = time.time()
        g_loss_total = 0.0
        d_loss_total = 0.0
        
        for i, (real_images, _) in enumerate(train_loader):
            real_images = real_images.to(device)
            batch_size = real_images.size(0)

            # Create labels
            real_labels = torch.ones(batch_size, 1).to(device)
            fake_labels = torch.zeros(batch_size, 1).to(device)

            try:
                # Train Discriminator
                optimizer_d.zero_grad()
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

                g_loss_total += g_loss.item()
                d_loss_total += d_loss.item()

            except (RuntimeError, ValueError) as e:
                logging.error(f"Error during training: {e}")
                continue  # Skip this batch if there's an error

        # Log losses
        avg_g_loss = g_loss_total / len(train_loader)
        avg_d_loss = d_loss_total / len(train_loader)
        epoch_time = time.time() - epoch_start_time

        logging.info(f'Epoch [{epoch+1}/{epochs}], Generator Loss: {avg_g_loss:.4f}, Discriminator Loss: {avg_d_loss:.4f}, Time: {epoch_time:.2f}s')

        # Store losses for visualization
        g_losses.append(avg_g_loss)
        d_losses.append(avg_d_loss)

        # Checkpointing
        checkpoint_dir = 'checkpoints'
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        torch.save(generator.state_dict(), os.path.join(checkpoint_dir, f'generator_epoch_{epoch+1}.pth'))
        torch.save(discriminator.state_dict(), os.path.join(checkpoint_dir, f'discriminator_epoch_{epoch+1}.pth'))

    logging.info("Training complete.")