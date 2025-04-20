import argparse
import yaml
import torch
import os
import sys
import logging
# Ensure project root (cwd) is on sys.path
sys.path.insert(0, os.getcwd())
# Also insert script directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model.generator import Generator
from utils.helpers import set_seed
from data.dataset import load_bookcorpus_dataset, get_tokenizer 
from training.trainer import Trainer

def main(config_path: str):
    """Main function to run the paraphrase generation training pipeline."""
    # --- 1. Load Configuration --- 
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: Configuration file not found at {config_path}")
        return
    except yaml.YAMLError as e:
        print(f"Error parsing configuration file: {e}")
        return

    # --- 2. Set Seed for Reproducibility --- 
    seed = config.get('random_seed', 42)
    set_seed(seed)

    # --- 3. Device Setup --- 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Download NLTK data ---
    import nltk, ssl
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context

    nltk.download('punkt')
    nltk.download('wordnet')

    # --- 4. Initialize Tokenizer (Placeholder) --- 
    # Replace with actual tokenizer loading based on config (e.g., from Hugging Face)
    tokenizer_name = config['data'].get('tokenizer_name', 'bert-base-uncased')
    tokenizer = get_tokenizer(tokenizer_name)
    vocab_size = tokenizer.vocab_size

    # --- 5. Load Data --- 
    data_config = config['data']
    # Load bookcorpus/wikitext dataset for language modeling
    original_num_workers = data_config.get('num_workers', 1)
    data_config['num_workers'] = 0
    train_loader, val_loader, test_loader = load_bookcorpus_dataset(data_config, tokenizer)

    # --- 6. Initialize Model (Generator Only for Language Modeling) --- 
    model_config = config['model']
    
    # Assuming embed_dim is derived from encoder/generator structure or specified
    # Example: use a common embed_dim
    generator = Generator(config=model_config, tokenizer=tokenizer)
    generator.to(device)

    # --- 7. Initialize Trainer (Needs Adaptation for LM) --- 
    trainer = Trainer(generator, tokenizer, config) # Pass only Generator

    # --- 8. Start Training (Needs Adaptation for LM) --- 
    trainer.train(train_loader, val_loader)

    # --- 9. Optional: Final Evaluation (Needs Adaptation for LM) --- 

    print("\nScript finished: Data loaded and Generator model initialized.")
    print("Next steps: Adapt the Trainer class or implement a new training loop for Language Modeling.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a VAE-InfoGAN Paraphrase Model.')
    parser.add_argument('--config', type=str, default='configs/default.yaml', 
                        help='Path to the configuration YAML file.')
    args = parser.parse_args()
    
    main(args.config)
