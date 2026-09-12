import os
import time
import torch
import logging
from torch.utils.data import DataLoader
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer, AdamW, get_linear_schedule_with_warmup
from datasets import load_dataset

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train(config: dict) -> None:
    """
    Train the DistilBERT model based on the provided configuration.
    
    Args:
        config (dict): Configuration settings for training.
    """
    # Load dataset
    dataset = load_dataset('imdb', split='train')
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

    # Tokenization
    def tokenize_function(examples):
        return tokenizer(examples['text'], padding="max_length", truncation=True)

    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    train_dataset = tokenized_dataset.shuffle(seed=42).select([i for i in list(range(10000))])  # For faster training

    # Initialize model
    model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Training parameters
    batch_size = config['global']['batch_size']['value']
    learning_rate = config['global']['learning_rate']['value']
    num_epochs = config['global']['num_epochs']['value']

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    total_steps = len(train_dataloader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        start_time = time.time()

        for batch in train_dataloader:
            optimizer.zero_grad()
            inputs = {key: val.to(device) for key, val in batch.items() if key in ['input_ids', 'attention_mask', 'labels']}
            outputs = model(**inputs)
            loss = outputs.loss
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
            scheduler.step()

        avg_loss = total_loss / len(train_dataloader)
        elapsed_time = time.time() - start_time
        logger.info(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}, Time: {elapsed_time:.2f}s")

        # Checkpointing
        if (epoch + 1) % 5 == 0:  # Save every 5 epochs
            checkpoint_path = f"checkpoints/model_epoch_{epoch + 1}.bin"
            torch.save(model.state_dict(), checkpoint_path)
            logger.info(f"Model checkpoint saved at {checkpoint_path}")

    # Save the final model
    final_model_path = "final_model.bin"
    torch.save(model.state_dict(), final_model_path)
    logger.info(f"Final model saved at {final_model_path}")

if __name__ == "__main__":
    # Example configuration
    config = {
        "global": {
            "batch_size": {"value": 32},
            "learning_rate": {"value": 5e-5},
            "num_epochs": {"value": 3}
        }
    }
    train(config)