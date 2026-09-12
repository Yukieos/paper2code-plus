import torch
import torch.nn as nn
import torch.optim as optim
import time
import os
from torch.utils.data import DataLoader

def train_model(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, optimizer: optim.Optimizer, criterion: nn.Module, epochs: int, device: str, checkpoint_dir: str) -> None:
    """
    Trains the TinyConvNet model using the provided data loader, optimizer, and loss function, logging metrics during training.

    Args:
        model (nn.Module): The TinyConvNet model to be trained.
        train_loader (DataLoader): DataLoader for training data.
        val_loader (DataLoader): DataLoader for validation data.
        optimizer (Optimizer): Optimizer for updating model weights.
        criterion (nn.Module): Loss function for training.
        epochs (int): Number of epochs to train.
        device (str): Device to run the training on ('cuda' or 'cpu').
        checkpoint_dir (str): Directory to save model checkpoints.
    """
    model.to(device)
    best_val_loss = float('inf')

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        start_time = time.time()

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if batch_idx % 10 == 0:  # Log every 10 batches
                print(f'Epoch [{epoch+1}/{epochs}], Step [{batch_idx}/{len(train_loader)}], Loss: {loss.item():.4f}')

        avg_loss = running_loss / len(train_loader)
        val_loss, val_accuracy = validate_model(model, val_loader, criterion, device)

        print(f'Epoch [{epoch+1}/{epochs}] completed in {time.time() - start_time:.2f}s. '
              f'Training Loss: {avg_loss:.4f}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%')

        # Checkpointing
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, optimizer, epoch, best_val_loss, checkpoint_dir)

def validate_model(model: nn.Module, val_loader: DataLoader, criterion: nn.Module, device: str) -> tuple:
    """
    Validates the model on the validation dataset.

    Args:
        model (nn.Module): The TinyConvNet model to be validated.
        val_loader (DataLoader): DataLoader for validation data.
        criterion (nn.Module): Loss function for validation.
        device (str): Device to run the validation on ('cuda' or 'cpu').

    Returns:
        tuple: Validation loss and accuracy.
    """
    model.eval()
    val_loss = 0.0
    correct = 0

    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            val_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    val_loss /= len(val_loader)
    accuracy = 100. * correct / len(val_loader.dataset)
    return val_loss, accuracy

def save_checkpoint(model: nn.Module, optimizer: optim.Optimizer, epoch: int, val_loss: float, checkpoint_dir: str) -> None:
    """
    Saves the model checkpoint.

    Args:
        model (nn.Module): The TinyConvNet model to be saved.
        optimizer (Optimizer): Optimizer state to be saved.
        epoch (int): Current epoch number.
        val_loss (float): Validation loss for the current epoch.
        checkpoint_dir (str): Directory to save the checkpoint.
    """
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    
    checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pth')
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss,
    }, checkpoint_path)
    print(f'Checkpoint saved at {checkpoint_path}')