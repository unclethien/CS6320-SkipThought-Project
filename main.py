import argparse
import yaml
import torch
import os
import sys
import logging
import nltk
import ssl

# Ensure project root (cwd) is on sys.path
sys.path.insert(0, os.getcwd())
# Also insert script directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.helpers import set_seed
from training.trainer import Trainer

def main(config_path: str):
    """Main function to run the Latent GAN training pipeline."""

    # --- Download NLTK data (for METEOR metric) ---
    try:
        logging.info("Checking/Downloading NLTK data (punkt, wordnet)...")
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass # Legacy Python versions may not have this
    else:
        # Handle potential SSL certificate issues during download
        ssl._create_default_https_context = _create_unverified_https_context
    
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError: # Catch LookupError when resource is not found
        logging.info("Downloading NLTK 'punkt' tokenizer...")
        nltk.download('punkt', quiet=True)
    try:
        nltk.data.find('corpora/wordnet')
    except LookupError: # Catch LookupError when resource is not found
        logging.info("Downloading NLTK 'wordnet' corpus...")
        nltk.download('wordnet', quiet=True)
    logging.info("NLTK data checks complete.")
    # ----------------------------------------------

    # --- 1. Load Configuration --- 
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            logging.info(f"Configuration loaded from {config_path}")
    except FileNotFoundError:
        logging.error(f"Error: Configuration file not found at {config_path}")
        return
    except yaml.YAMLError as e:
        logging.error(f"Error parsing configuration file: {e}")
        return

    # --- 2. Set Seed for Reproducibility --- 
    seed = config.get('seed', 42) 
    set_seed(seed)
    logging.info(f"Random seed set to {seed}")

    # --- 4. Initialize Trainer --- 
    logging.info("Initializing Trainer...")
    try:
        trainer = Trainer(config_path=config_path)
    except Exception as e:
        logging.error(f"Error initializing Trainer: {e}", exc_info=True)
        return
    logging.info("Trainer initialized successfully.")

    # --- 5. Start Training --- 
    logging.info("Starting training process...")
    try:
        trainer.train()
        logging.info("Training finished successfully.")
    except Exception as e:
        logging.error(f"Error during training: {e}", exc_info=True)

    logging.info("\nScript finished.")

if __name__ == '__main__':
    # --- Setup Logging --- 
    log_level_str = os.environ.get("LOG_LEVEL", "INFO").upper()
    log_level = getattr(logging, log_level_str, logging.INFO)
    logging.basicConfig(level=log_level,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        handlers=[logging.StreamHandler(sys.stdout)]) 

    parser = argparse.ArgumentParser(description='Train a Transformer-Based Latent GAN.') 
    parser.add_argument('--config', type=str, default='configs/default.yaml', 
                        help='Path to the configuration YAML file.')
    args = parser.parse_args()
    
    main(args.config)
