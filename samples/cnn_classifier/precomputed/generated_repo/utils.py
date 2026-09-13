import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from model import train_model, validate_model  # Importing train_model and validate_model from model

def initialize_optimizer(model, learning_rate):
    return optim.Adam(model.parameters(), lr=learning_rate)

def initialize_scheduler(optimizer, step_size, gamma):
    return optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def save_model(model, file_path):
    """Save the model to the specified file path."""
    try:
        torch.save(model.state_dict(), file_path)
    except Exception as e:
        print(f"Error saving the model: {e}")

def load_model(file_path, model):
    """Load the model from the specified file path."""
    try:
        model.load_state_dict(torch.load(file_path))
        model.eval()  # Set the model to evaluation mode
        return model
    except FileNotFoundError:
        print(f"Error: The file {file_path} was not found.")
    except RuntimeError as e:
        print(f"Error loading the model: {e}")
    return None