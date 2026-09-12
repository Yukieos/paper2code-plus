import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import time
import os

class Generator(nn.Module):
    """Generator class for MiniGAN that maps latent vectors to images."""
    
    def __init__(self, latent_dim, output_dim):
        super(Generator, self).__init__()
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        
        self.model = nn.Sequential(
            nn.Linear(self.latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, self.output_dim),
            nn.Tanh()
        )
    
    def forward(self, z):
        return self.model(z)

class Discriminator(nn.Module):
    """Discriminator class for MiniGAN that maps images to real/fake probabilities."""
    
    def __init__(self, input_dim):
        super(Discriminator, self).__init__()
        self.input_dim = input_dim
        
        self.model = nn.Sequential(
            nn.Linear(self.input_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.model(x)

def train_model(generator, discriminator, dataloader, epochs, learning_rate, log_interval, device):
    criterion = nn.BCELoss()
    optimizer_g = optim.Adam(generator.parameters(), lr=learning_rate)
    optimizer_d = optim.Adam(discriminator.parameters(), lr=learning_rate)
    
    generator.to(device)
    discriminator.to(device)

    for epoch in range(epochs):
        start_time = time.time()
        for i, (images, _) in enumerate(dataloader):
            batch_size = images.size(0)
            images = images.view(batch_size, -1).to(device)
            real_labels = torch.ones(batch_size, 1).to(device)
            fake_labels = torch.zeros(batch_size, 1).to(device)

            # Train Discriminator
            optimizer_d.zero_grad()
            outputs = discriminator(images)
            d_loss_real = criterion(outputs, real_labels)
            d_loss_real.backward()

            z = torch.randn(batch_size, generator.latent_dim).to(device)
            fake_images = generator(z)
            outputs = discriminator(fake_images.detach())
            d_loss_fake = criterion(outputs, fake_labels)
            d_loss_fake.backward()
            optimizer_d.step()

            # Train Generator
            optimizer_g.zero_grad()
            outputs = discriminator(fake_images)
            g_loss = criterion(outputs, real_labels)
            g_loss.backward()
            optimizer_g.step()

            if (i + 1) % log_interval == 0:
                print(f'Epoch [{epoch + 1}/{epochs}], Step [{i + 1}/{len(dataloader)}], '
                      f'D Loss: {d_loss_real.item() + d_loss_fake.item():.4f}, '
                      f'G Loss: {g_loss.item():.4f}, Time: {time.time() - start_time:.2f}s')

        # Save model checkpoints
        if (epoch + 1) % 10 == 0:
            torch.save(generator.state_dict(), f'generator_epoch_{epoch + 1}.pth')
            torch.save(discriminator.state_dict(), f'discriminator_epoch_{epoch + 1}.pth')

def main():
    # Configuration
    batch_size = 64
    epochs = 100
    learning_rate = 0.0002
    log_interval = 100
    latent_dim = 100
    output_dim = 28 * 28  # 28x28 images flattened

    # Data loading
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Model initialization
    generator = Generator(latent_dim, output_dim)
    discriminator = Discriminator(output_dim)

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Train the model
    train_model(generator, discriminator, dataloader, epochs, learning_rate, log_interval, device)

if __name__ == '__main__':
    main()