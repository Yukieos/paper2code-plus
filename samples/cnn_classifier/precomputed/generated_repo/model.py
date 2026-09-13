import torch
import torch.nn as nn
import torch.optim as optim
import time
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class TinyConvNet(nn.Module):
    """Defines the TinyConvNet architecture for digit classification."""
    
    def __init__(self, input_shape):
        super(TinyConvNet, self).__init__()
        
        # Validate input shape
        if not isinstance(input_shape, tuple) or len(input_shape) != 3 or input_shape[0] != 1:
            raise ValueError("input_shape must be a tuple of the form (1, height, width) where height and width are positive integers.")
        
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.fc = nn.Linear(32 * (input_shape[1] // 4) * (input_shape[2] // 4), 10)  # Adjusted for pooling

    def forward(self, x):
        # Validate input shape
        if x.dim() != 4 or x.size(1) != 1:
            raise ValueError("Input tensor must have shape (batch_size, 1, height, width).")
        
        x = nn.functional.relu(self.conv1(x))
        x = nn.functional.max_pool2d(x, 2)  # 14x14
        x = nn.functional.relu(self.conv2(x))
        x = nn.functional.max_pool2d(x, 2)  # 7x7
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc(x)
        return x

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device, checkpoint_path):
    model.to(device)
    best_val_loss = float('inf')

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
        val_loss = validate_model(model, val_loader, criterion, device)

        logging.info(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}, Val Loss: {val_loss:.4f}, Time: {time.time() - start_time:.2f}s')

        # Checkpointing
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), checkpoint_path)
            logging.info(f'Model checkpoint saved at epoch {epoch + 1}')

def validate_model(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()

    return running_loss / len(val_loader)

def main():
    # Configuration
    learning_rate = 0.001
    batch_size = 64
    num_epochs = 5
    checkpoint_path = 'tinyconvnet_checkpoint.pth'
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize model, criterion, optimizer
    input_shape = (1, 28, 28)  # Example input shape for grayscale images
    model = TinyConvNet(input_shape)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Load your data here
    # train_loader, val_loader = load_data(batch_size)

    # Train the model
    train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device, checkpoint_path)

if __name__ == '__main__':
    main()