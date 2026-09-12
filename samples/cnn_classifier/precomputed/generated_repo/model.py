import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import time
import logging
import os

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class TinyConvNet(nn.Module):
    """TinyConvNet model for digit classification."""
    
    def __init__(self, input_channels: int = 1, num_classes: int = 10):
        super(TinyConvNet, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)  # Flatten the tensor
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def train_model(model, train_loader, val_loader, criterion, optimizer, epochs, device, checkpoint_path):
    model.to(device)
    for epoch in range(epochs):
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
        logging.info(f'Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss:.4f}, Time: {time.time() - start_time:.2f}s')
        
        # Validation
        validate_model(model, val_loader, criterion, device)

        # Checkpointing
        if (epoch + 1) % 5 == 0:  # Save every 5 epochs
            save_checkpoint(model, optimizer, epoch, checkpoint_path)

def validate_model(model, val_loader, criterion, device):
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
    logging.info(f'Validation Loss: {avg_val_loss:.4f}, Accuracy: {accuracy:.2f}%')

def save_checkpoint(model, optimizer, epoch, checkpoint_path):
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    checkpoint_file = os.path.join(checkpoint_path, f'checkpoint_epoch_{epoch + 1}.pth')
    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, checkpoint_file)
    logging.info(f'Checkpoint saved: {checkpoint_file}')

def main():
    # Configuration
    learning_rate = 1e-3
    batch_size = 64
    epochs = 5
    checkpoint_path = './checkpoints'
    
    # Prepare dataset (this is a placeholder, replace with actual dataset loading)
    train_dataset = ...  # Load your training dataset here
    val_dataset = ...    # Load your validation dataset here
    train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Initialize model, criterion, and optimizer
    model = TinyConvNet()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Train the model
    train_model(model, train_loader, val_loader, criterion, optimizer, epochs, device, checkpoint_path)

if __name__ == '__main__':
    main()