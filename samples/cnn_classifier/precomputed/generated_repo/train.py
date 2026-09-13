import torch
import torch.nn as nn
import torch.optim as optim
import logging
import time
from torch.utils.data import DataLoader
from typing import Callable

def train(model: nn.Module, train_loader: DataLoader, optimizer: optim.Optimizer, criterion: Callable, epochs: int, device: str, scheduler: optim.lr_scheduler._LRScheduler = None) -> None:
    """
    Trains the TinyConvNet model on the training dataset, with error handling and logging.

    Args:
        model (nn.Module): The TinyConvNet model instance.
        train_loader (DataLoader): DataLoader for training data.
        optimizer (Optimizer): Optimizer for training the model.
        criterion (Callable): Loss function for training.
        epochs (int): Number of epochs for training.
        device (str): Device to run the training on ('cuda' or 'cpu').
        scheduler (optim.lr_scheduler._LRScheduler, optional): Learning rate scheduler.
    """
    logger = logging.getLogger(__name__)
    model.to(device)
    model.train()

    for epoch in range(epochs):
        epoch_start_time = time.time()
        total_loss = 0.0
        correct = 0
        total = 0

        try:
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(device), target.to(device)

                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()

            if scheduler:
                scheduler.step()

            epoch_loss = total_loss / len(train_loader)
            epoch_accuracy = 100. * correct / total
            epoch_end_time = time.time()

            logger.info(f'Epoch: {epoch + 1}/{epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%, Time: {epoch_end_time - epoch_start_time:.2f}s')

            # Checkpointing
            if (epoch + 1) % 5 == 0:  # Save every 5 epochs
                checkpoint_path = f'checkpoint_epoch_{epoch + 1}.pth'
                torch.save(model.state_dict(), checkpoint_path)
                logger.info(f'Model checkpoint saved to {checkpoint_path}')

        except RuntimeError as e:
            logger.error(f'RuntimeError during training: {e}')
            break

    logger.info('Training complete.')