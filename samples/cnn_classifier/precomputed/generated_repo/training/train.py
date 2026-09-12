import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time
import os
import logging

# Configure logger outside the function to avoid multiple configurations
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define device, log_interval, and save_path
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
log_interval = 10
save_path = 'checkpoints'

def train_model(model: nn.Module, train_loader: DataLoader, optimizer: optim.Optimizer, criterion: callable, epochs: int) -> (float, float):
    """
    Trains the TinyConvNet model using the provided data loader, optimizer, and loss function, with error handling for training process.

    Args:
        model (nn.Module): The TinyConvNet model instance.
        train_loader (DataLoader): Data loader for training data.
        optimizer (Optimizer): Optimizer for training the model.
        criterion (callable): Loss function for training.
        epochs (int): Number of epochs for training.

    Returns:
        tuple: Average training loss and accuracy after training.
    """
    model.to(device)
    model.train()

    train_loss = 0.0
    train_accuracy = 0.0

    try:
        for epoch in range(epochs):
            epoch_start_time = time.time()
            running_loss = 0.0
            correct = 0
            total = 0

            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(device), target.to(device)

                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()

                if batch_idx % log_interval == 0:
                    logger.info(f'Epoch: {epoch + 1}/{epochs} [{batch_idx * len(data)}/{len(train_loader.dataset)}] Loss: {loss.item():.6f}')

            epoch_loss = running_loss / len(train_loader)
            epoch_accuracy = correct / total
            epoch_time = time.time() - epoch_start_time

            logger.info(f'Epoch {epoch + 1}/{epochs} - Loss: {epoch_loss:.6f}, Accuracy: {epoch_accuracy:.4f}, Time: {epoch_time:.2f}s')

            # Save checkpoint
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            torch.save(model.state_dict(), os.path.join(save_path, f'model_epoch_{epoch + 1}.pth'))

            train_loss += epoch_loss
            train_accuracy += epoch_accuracy

    except (RuntimeError, ValueError) as e:
        logger.error(f'Error during training: {str(e)}')

    train_loss /= epochs
    train_accuracy /= epochs

    logger.info(f'Final Average Training Loss: {train_loss:.6f}, Average Accuracy: {train_accuracy:.4f}')
    logger.info('Training complete.')
    return train_loss, train_accuracy