import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import time
import os

class Generator(nn.Module):
    """Generator model for MiniGAN. 

    Maps a latent vector to a 28x28 grayscale image.
    """
    def __init__(self, latent_vector_size):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(latent_vector_size, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 28 * 28)
        
    def forward(self, z):
        x = torch.relu(self.fc1(z))
        x = torch.relu(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        return x.view(-1, 1, 28, 28)

class Discriminator(nn.Module):
    """Discriminator model for MiniGAN. 

    Maps a 28x28 grayscale image to a real/fake probability.
    """
    def __init__(self):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 1)
        
    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = torch.leaky_relu(self.fc1(x), 0.2)
        x = torch.leaky_relu(self.fc2(x), 0.2)
        return torch.sigmoid(self.fc3(x))

def train(generator, discriminator, dataloader, num_epochs, learning_rate, device, checkpoint_dir):
    criterion = nn.BCELoss()
    optimizer_g = optim.Adam(generator.parameters(), lr=learning_rate)
    optimizer_d = optim.Adam(discriminator.parameters(), lr=learning_rate)
    
    generator.train()
    discriminator.train()
    
    for epoch in range(num_epochs):
        start_time = time.time()
        for i, (real_images, _) in enumerate(dataloader):
            real_images = real_images.to(device)
            batch_size = real_images.size(0)
            
            # Create labels
            real_labels = torch.ones(batch_size, 1).to(device)
            fake_labels = torch.zeros(batch_size, 1).to(device)
            
            # Train Discriminator
            optimizer_d.zero_grad()
            outputs = discriminator(real_images)
            d_loss_real = criterion(outputs, real_labels)
            d_loss_real.backward()
            
            z = torch.randn(batch_size, 100).to(device)
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
            
            if (i + 1) % 100 == 0:
                print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(dataloader)}], '
                      f'D Loss: {d_loss_real.item() + d_loss_fake.item():.4f}, G Loss: {g_loss.item():.4f}')
        
        # Save checkpoint
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        torch.save(generator.state_dict(), os.path.join(checkpoint_dir, f'generator_epoch_{epoch + 1}.pth'))
        torch.save(discriminator.state_dict(), os.path.join(checkpoint_dir, f'discriminator_epoch_{epoch + 1}.pth'))
        
        end_time = time.time()
        print(f'Epoch [{epoch + 1}/{num_epochs}] completed in {end_time - start_time:.2f} seconds.')

def main():
    # Configuration
    batch_size = 64
    num_epochs = 50
    learning_rate = 0.0002
    latent_vector_size = 100
    checkpoint_dir = './checkpoints'
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Initialize models
    generator = Generator(latent_vector_size).to(device)
    discriminator = Discriminator().to(device)
    
    # Start training
    train(generator, discriminator, dataloader, num_epochs, learning_rate, device, checkpoint_dir)

if __name__ == '__main__':
    main()