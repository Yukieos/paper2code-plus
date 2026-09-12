import argparse
import json
import logging
import os
import random
import pandas as pd
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info("Logging is set up.")

def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def load_data(batch_size: int) -> (DataLoader, DataLoader):
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader

class TinyConvNet(nn.Module):
    def __init__(self):
        super(TinyConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.fc = nn.Linear(32 * 7 * 7, 10)
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = self.pool(nn.functional.relu(self.conv2(x)))
        x = x.view(-1, 32 * 7 * 7)
        x = self.fc(x)
        return x

def train_model(model: nn.Module, train_loader: DataLoader, learning_rate: float, epochs: int) -> None:
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        logging.info(f'Epoch [{epoch + 1}/{epochs}], Loss: {running_loss / len(train_loader):.4f}')

def evaluate_model(model: nn.Module, test_loader: DataLoader) -> dict:
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    logging.info(f'Accuracy of the model on the test images: {accuracy:.2f}%')
    return {'accuracy': accuracy}

def report_results(results: dict, output_path: str) -> None:
    df = pd.DataFrame([results])
    df.to_csv(output_path, index=False)
    logging.info(f'Results saved to {output_path}')

def visualize_results(results: dict) -> None:
    plt.figure(figsize=(10, 5))
    plt.bar(['Accuracy'], [results['accuracy']], color='blue')
    plt.ylim(0, 100)
    plt.ylabel('Accuracy (%)')
    plt.title('Model Evaluation Results')
    plt.savefig('evaluation_results.png')
    plt.show()

def main() -> None:
    parser = argparse.ArgumentParser(description='TinyConvNet Evaluation')
    parser.add_argument('--config', type=str, default='config.json', help='Path to configuration file')
    parser.add_argument('--seed', type=int, default=None, help='Random seed for reproducibility')
    args = parser.parse_args()

    setup_logging()
    config = load_config(args.config)

    if args.seed is not None:
        set_seed(args.seed)

    learning_rate = config['global']['learning_rate']['value']
    batch_size = config['global']['batch_size']['value']
    epochs = config['global']['epochs']['value']

    train_loader, test_loader = load_data(batch_size)
    model = TinyConvNet()
    
    train_model(model, train_loader, learning_rate, epochs)
    results = evaluate_model(model, test_loader)
    report_results(results, 'evaluation_results.csv')
    visualize_results(results)

if __name__ == '__main__':
    main()