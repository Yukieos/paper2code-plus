import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import DistilBertModel, DistilBertTokenizer
import time
import os

class DistilBERT(nn.Module):
    """
    DistilBERT model architecture based on the Transformer architecture.
    """
    def __init__(self):
        super(DistilBERT, self).__init__()
        self.distilbert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        
    def forward(self, inputs):
        outputs = self.distilbert(inputs)
        return outputs.logits

def train_model(model, train_loader, val_loader, num_epochs, learning_rate, device, checkpoint_dir):
    model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        start_time = time.time()
        
        for batch in train_loader:
            inputs, labels = batch['input_ids'].to(device), batch['labels'].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = nn.CrossEntropyLoss()(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}, Time: {time.time() - start_time:.2f}s')
        
        validate_model(model, val_loader, device)
        save_checkpoint(model, optimizer, epoch, checkpoint_dir)

        scheduler.step()

def validate_model(model, val_loader, device):
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch in val_loader:
            inputs, labels = batch['input_ids'].to(device), batch['labels'].to(device)
            outputs = model(inputs)
            loss = nn.CrossEntropyLoss()(outputs, labels)
            total_loss += loss.item()
    
    avg_loss = total_loss / len(val_loader)
    print(f'Validation Loss: {avg_loss:.4f}')

def save_checkpoint(model, optimizer, epoch, checkpoint_dir):
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    
    checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch + 1}.pt')
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, checkpoint_path)
    print(f'Checkpoint saved at {checkpoint_path}')