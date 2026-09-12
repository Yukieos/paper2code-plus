import torch
import torch.nn as nn
import torch.optim as optim
import time
import os
import logging

class Generator(nn.Module):
    """Generator model for MiniGAN that maps latent vectors to images, with specified layer configurations and activation functions."""
    
    def __init__(self, latent_dim, img_dim):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(latent_dim, 128)
        self.fc2 = nn.Linear(128, img_dim)
        self.activation = nn.ReLU()

    def forward(self, z):
        x = self.activation(self.fc1(z))
        return torch.tanh(self.fc2(x))


class Discriminator(nn.Module):
    """Discriminator model for MiniGAN that maps images to real/fake probabilities, with specified layer configurations and activation functions."""
    
    def __init__(self, img_dim):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(img_dim, 128)
        self.fc2 = nn.Linear(128, 1)
        self.activation = nn.LeakyReLU(0.2)

    def forward(self, x):
        x = self.activation(self.fc1(x))
        return torch.sigmoid(self.fc2(x))


def train_model(generator, discriminator, train_loader, epochs, learning_rate, device, checkpoint_dir):
    """Train the MiniGAN model using the defined loss function and optimizer."""
    
    # Move models to device
    generator.to(device)
    discriminator.to(device)

    # Initialize optimizers
    optimizer_g = optim.Adam(generator.parameters(), lr=learning_rate)
    optimizer_d = optim.Adam(discriminator.parameters(), lr=learning_rate)

    # Loss function
    criterion = nn.BCELoss()

    # Logging setup
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    for epoch in range(epochs):
        start_time = time.time()
        generator.train()
        discriminator.train()

        for i, (real_images, _) in enumerate(train_loader):
            real_images = real_images.view(real_images.size(0), -1).to(device)
            batch_size = real_images.size(0)

            # Create labels
            real_labels = torch.ones(batch_size, 1).to(device)
            fake_labels = torch.zeros(batch_size, 1).to(device)

            # Train Discriminator
            optimizer_d.zero_grad()
            outputs = discriminator(real_images)
            d_loss_real = criterion(outputs, real_labels)

            z = torch.randn(batch_size, latent_dim).to(device)
            fake_images = generator(z)
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

            if (i + 1) % 100 == 0:
                logger.info(f'Epoch [{epoch + 1}/{epochs}], Step [{i + 1}/{len(train_loader)}], '
                            f'D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}')

        # Save checkpoint
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        torch.save(generator.state_dict(), os.path.join(checkpoint_dir, f'generator_epoch_{epoch + 1}.pth'))
        torch.save(discriminator.state_dict(), os.path.join(checkpoint_dir, f'discriminator_epoch_{epoch + 1}.pth'))

        elapsed_time = time.time() - start_time
        logger.info(f'Epoch [{epoch + 1}/{epochs}] completed in {elapsed_time:.2f} seconds.')

    logger.info('Training completed.')