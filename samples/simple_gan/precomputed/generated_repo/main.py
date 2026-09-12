import json
import sys
from data_loader import HandwrittenDigitDataset
from evaluation import Generator, load_checkpoint, calculate_fid, plot_loss_curves, evaluate
from models import Generator as ModelGenerator, Discriminator, train_model
from training import train
from utils import set_seed, save_model

def main(config_path: str, seed: int) -> None:
    """Main entry point for the MiniGAN training and evaluation process. Accepts a configuration file and a random seed for reproducibility."""
    
    # Set random seed for reproducibility
    set_seed(seed)

    # Load configuration from the specified JSON file
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
    except FileNotFoundError:
        print(f"Error: Configuration file '{config_path}' not found.")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error: Configuration file '{config_path}' is not a valid JSON.")
        sys.exit(1)

    # Validate configuration contents
    required_keys = ['global', 'epochs', 'learning_rate', 'log_interval']
    if not all(key in config for key in required_keys):
        print("Error: Configuration file is missing required keys.")
        sys.exit(1)

    # Initialize the dataset and data loaders
    dataset = HandwrittenDigitDataset(batch_size=config['global']['batch_size']['value'])
    data_loader = dataset.load_data()

    # Create the MiniGAN model
    latent_dim = 100  # Example latent dimension
    output_dim = 28 * 28  # Output dimension for 28x28 images
    try:
        generator = ModelGenerator(latent_dim, output_dim)
        discriminator = Discriminator(input_dim=output_dim)
    except Exception as e:
        print(f"Error initializing models: {e}")
        sys.exit(1)

    # Train the model using the training loop
    try:
        train(generator, discriminator, data_loader, 
              epochs=config['global']['epochs']['value'], 
              learning_rate=config['global']['learning_rate']['value'], 
              log_interval=config['global']['log_interval']['value'])
    except KeyboardInterrupt:
        print("Training interrupted. Saving model state...")
        save_model(generator, 'generator_interrupted.pth')
        save_model(discriminator, 'discriminator_interrupted.pth')
        sys.exit(1)
    except Exception as e:
        print(f"Error during training: {e}")
        sys.exit(1)

    # Evaluate the model and report results
    evaluate(generator, data_loader, output_dir='output')

if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else 'config.json'
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 42
    main(config_path, seed)