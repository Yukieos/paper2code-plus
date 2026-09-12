import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time
import os
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_model(model: nn.Module, train_loader: DataLoader, criterion: nn.Module, optimizer: optim.Optimizer, epochs: int) -> None:
    """
    Train the TinyConvNet model.

    Args:
        model (nn.Module): The TinyConvNet model.
        train_loader (DataLoader): DataLoader for training data.
        criterion (nn.Module): Loss function.
        optimizer (Optimizer): Optimizer for updating model weights.
        epochs (int): Number of epochs to train.

    Returns:
        None
    """
    model.to(device)  # Move model to the specified device
    model.train()
    
    if len(train_loader) == 0:
        logging.error('Training data is empty. Exiting training.')
        return

    metrics = {'average_loss': 0, 'accuracy': 0}

    for epoch in range(epochs):
        epoch_start_time = time.time()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)

            try:
                # Zero the gradients
                optimizer.zero_grad()

                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, targets)

                # Check for NaN values in loss
                if torch.isnan(loss):
                    logging.error(f'NaN loss encountered at epoch {epoch + 1}, batch {batch_idx + 1}. Skipping update.')
                    continue

                # Backward pass and optimize
                loss.backward()
                optimizer.step()

                # Statistics
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()

                if batch_idx % 10 == 0:  # Log every 10 batches
                    logging.info(f'Epoch [{epoch + 1}/{epochs}], Step [{batch_idx + 1}/{len(train_loader)}], Loss: {loss.item():.4f}')

            except Exception as e:
                logging.error(f'Error during training at epoch {epoch + 1}, batch {batch_idx + 1}: {str(e)}')

        # Calculate average loss and accuracy
        avg_loss = running_loss / len(train_loader)
        accuracy = 100 * correct / total if total > 0 else 0
        epoch_duration = time.time() - epoch_start_time

        logging.info(f'Epoch [{epoch + 1}/{epochs}] completed in {epoch_duration:.2f} seconds. '
                     f'Average Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%')

        # Save checkpoint
        save_path = 'checkpoints'
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        torch.save(model.state_dict(), os.path.join(save_path, f'model_epoch_{epoch + 1}.pth'))

        # Update metrics
        metrics['average_loss'] += avg_loss
        metrics['accuracy'] += accuracy

    # Average metrics over epochs
    metrics['average_loss'] /= epochs
    metrics['accuracy'] /= epochs

    logging.info(f'Training completed. Final Average Loss: {metrics["average_loss"]:.4f}, Final Accuracy: {metrics["accuracy"]:.2f}%')