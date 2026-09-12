import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time
import os
import logging

# Define the TinyConvNet model architecture
class TinyConvNet(nn.Module):
    """
    Defines the TinyConvNet architecture for digit classification.
    """
    def __init__(self):
        super(TinyConvNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.fc = nn.Linear(32 * 7 * 7, 10)  # Assuming input size is 28x28

    def forward(self, x):
        x = nn.functional.relu(self.conv1(x))
        x = nn.functional.max_pool2d(x, kernel_size=2, stride=2)
        x = nn.functional.relu(self.conv2(x))
        x = nn.functional.max_pool2d(x, kernel_size=2, stride=2)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.fc(x)
        return x

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device, checkpoint_dir):
    logging.info("Starting training...")
    model.to(device)
    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        start_time = time.time()

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        val_loss = validate_model(model, val_loader, criterion, device)

        logging.info(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}, Val Loss: {val_loss:.4f}, Time: {time.time() - start_time:.2f}s')

        # Checkpointing
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, optimizer, epoch, val_loss, checkpoint_dir)

def validate_model(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0.0

    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            val_loss += loss.item()

    return val_loss / len(val_loader)

def save_checkpoint(model, optimizer, epoch, val_loss, checkpoint_dir):
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch + 1}.pth')
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss,
    }, checkpoint_path)
    logging.info(f'Checkpoint saved at {checkpoint_path}')

def main():
    # Setup logging
    logging.basicConfig(level=logging.INFO)

    # Load configuration settings
    learning_rate = 1e-3
    batch_size = 64
    num_epochs = 5
    checkpoint_dir = './checkpoints'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load dataset (placeholder, replace with actual dataset loading)
    train_loader = DataLoader(...)  # Replace with actual DataLoader
    val_loader = DataLoader(...)  # Replace with actual DataLoader

    # Initialize model, criterion, and optimizer
    model = TinyConvNet()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model
    train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device, checkpoint_dir)

if __name__ == "__main__":
    main()