import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer

def train(model: nn.Module, train_loader: DataLoader, optimizer: Optimizer, criterion: nn.Module, epochs: int) -> None:
    """
    Train the MiniEncoder model on the training dataset.

    Args:
        model (nn.Module): The MiniEncoder model to train.
        train_loader (DataLoader): DataLoader for the training dataset.
        optimizer (Optimizer): Optimizer for updating model weights.
        criterion (nn.Module): Loss function for training.
        epochs (int): Number of training epochs.

    Raises:
        ValueError: If epochs is not a positive integer.
        TypeError: If model, train_loader, optimizer, or criterion are of incorrect type.
    """
    
    # Input validation
    if not isinstance(model, nn.Module):
        raise TypeError("Expected model to be an instance of nn.Module.")
    if not isinstance(train_loader, DataLoader):
        raise TypeError("Expected train_loader to be an instance of DataLoader.")
    if not isinstance(optimizer, Optimizer):
        raise TypeError("Expected optimizer to be an instance of Optimizer.")
    if not isinstance(criterion, nn.Module):
        raise TypeError("Expected criterion to be an instance of nn.Module.")
    if not isinstance(epochs, int) or epochs <= 0:
        raise ValueError("Epochs must be a positive integer.")

    # Set the model to training mode
    model.train()

    for epoch in range(epochs):
        epoch_loss = 0.0  # Initialize loss for the epoch

        for batch in train_loader:
            inputs, targets = batch  # Unpack the batch
            
            # Validate inputs and targets
            if not isinstance(inputs, torch.Tensor) or not isinstance(targets, torch.Tensor):
                raise TypeError("Inputs and targets must be of type torch.Tensor.")
            if inputs.ndim != 2 or targets.ndim != 1:
                raise ValueError("Inputs must be 2D and targets must be 1D.")

            # Move inputs and targets to the appropriate device (GPU or CPU)
            device = inputs.device  # Use the device of the inputs
            inputs, targets = inputs.to(device), targets.to(device)

            try:
                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, targets)  # Compute loss

                # Backward pass and optimization
                optimizer.zero_grad()  # Clear previous gradients
                loss.backward()  # Backpropagation
                optimizer.step()  # Update model weights

                epoch_loss += loss.item()  # Accumulate loss

            except RuntimeError as e:
                if 'out of memory' in str(e):
                    print("Out of memory error. Reducing batch size or model size may help.")
                    return
                else:
                    raise  # Re-raise unexpected errors

        # Print epoch loss for monitoring
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {epoch_loss / len(train_loader):.4f}")

    print("Training complete.")  # Indicate completion of training