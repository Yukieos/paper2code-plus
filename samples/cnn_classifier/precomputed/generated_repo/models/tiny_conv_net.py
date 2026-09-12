import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class TinyConvNet(nn.Module):
    """Defines the TinyConvNet architecture for digit classification."""
    
    def __init__(self):
        super(TinyConvNet, self).__init__()
        # Initialize convolutional layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        # Initialize fully connected layers
        self.fc = nn.Linear(32 * 7 * 7, 10)  # Assuming input size is 28x28

    def forward(self, x):
        """Defines the forward pass."""
        x = nn.functional.relu(self.conv1(x))
        x = nn.functional.max_pool2d(x, kernel_size=2, stride=2)
        x = nn.functional.relu(self.conv2(x))
        x = nn.functional.max_pool2d(x, kernel_size=2, stride=2)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc(x)
        return x

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device):
    """Train the model with the given parameters."""
    model.to(device)
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        start_time = time.time()
        
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        avg_loss = running_loss / len(train_loader)
        val_loss, val_accuracy = validate_model(model, val_loader, criterion, device)
        elapsed_time = time.time() - start_time
        
        logging.info(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, '
                     f'Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%, '
                     f'Time: {elapsed_time:.2f}s')
        
        # Save checkpoint
        if (epoch + 1) % 5 == 0:  # Save every 5 epochs
            save_checkpoint(model, optimizer, epoch + 1)

def validate_model(model, val_loader, criterion, device):
    """Validate the model on the validation set."""
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    avg_val_loss = val_loss / len(val_loader)
    accuracy = 100 * correct / total
    return avg_val_loss, accuracy

def save_checkpoint(model, optimizer, epoch):
    """Save the model checkpoint."""
    checkpoint_path = 'checkpoints'
    os.makedirs(checkpoint_path, exist_ok=True)
    checkpoint_file = os.path.join(checkpoint_path, f'checkpoint_epoch_{epoch}.pth')
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, checkpoint_file)
    logging.info(f'Checkpoint saved at {checkpoint_file}')

def main():
    # Configuration settings
    learning_rate = 0.001
    batch_size = 64
    num_epochs = 5
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load dataset (placeholder, replace with actual dataset loading)
    train_dataset = ...  # Load your training dataset here
    val_dataset = ...    # Load your validation dataset here
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize model, criterion, and optimizer
    model = TinyConvNet()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Train the model
    train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device)

if __name__ == '__main__':
    main()