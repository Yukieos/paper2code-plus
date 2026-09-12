import torch
import torch.optim as optim
import time
import logging
from torch.utils.data import DataLoader

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def train(generator, discriminator, data_loader: DataLoader, epochs: int, learning_rate: float, log_interval: int) -> None:
    """
    Train the MiniGAN model using the specified data loader and parameters, with logging.

    Args:
        generator: The generator model.
        discriminator: The discriminator model.
        data_loader: Data loader for training data.
        epochs: Number of training epochs.
        learning_rate: Learning rate for the optimizer.
        log_interval: Interval for logging training progress.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator.to(device)
    discriminator.to(device)

    # Set up optimizers
    optimizer_g = optim.Adam(generator.parameters(), lr=learning_rate)
    optimizer_d = optim.Adam(discriminator.parameters(), lr=learning_rate)

    # Loss function
    criterion = torch.nn.BCELoss()

    # Training loop
    for epoch in range(epochs):
        generator.train()
        discriminator.train()
        start_time = time.time()
        
        for batch_idx, (real_images, _) in enumerate(data_loader):
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

            # Logging
            if batch_idx % log_interval == 0:
                logging.info(f'Epoch [{epoch+1}/{epochs}], Step [{batch_idx}/{len(data_loader)}], '
                             f'D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}')

        # Checkpointing
        torch.save(generator.state_dict(), f'generator_epoch_{epoch+1}.pth')
        torch.save(discriminator.state_dict(), f'discriminator_epoch_{epoch+1}.pth')

        end_time = time.time()
        logging.info(f'Epoch [{epoch+1}/{epochs}] completed in {end_time - start_time:.2f} seconds.')

    logging.info('Training completed.')