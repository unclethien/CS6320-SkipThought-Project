import torch
import torch.optim as optim
from torch.amp import GradScaler, autocast
import os
import yaml
from tqdm import tqdm
import math
import logging
import numpy as np

# Import only the Generator model
from model.generator import Generator 

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class Trainer:
    """Handles the training loop for a standard Language Model (Generator)."""
    def __init__(self, generator: Generator, tokenizer, config: dict):
        self.generator = generator
        self.config = config
        self.tokenizer = tokenizer
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Move models to device
        self.generator.to(self.device)

        # Optimizer (AdamW recommended for Transformers)
        lr = config['training'].get('learning_rate', 1e-4) # Single learning rate
        self.optimizer = optim.AdamW(self.generator.parameters(), lr=lr)

        # Loss function for Language Modeling
        # Ignore padding index if tokenizer has one
        pad_token_id = tokenizer.pad_token_id if hasattr(tokenizer, 'pad_token_id') else -100
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=pad_token_id)

        # Mixed Precision Scaler
        self.use_amp = config['training'].get('use_amp', False) and self.device.type == 'cuda'
        self.scaler = torch.amp.GradScaler('cuda', enabled=self.use_amp) if self.device.type == 'cuda' else None # Correct initialization

        # Training settings
        self.num_epochs = config['training'].get('num_epochs', 10) # Adjust default epochs
        self.early_stopping_patience = config['training'].get('early_stopping_patience', 3)
        self.patience = config['training'].get('patience', 5)

        self.output_dir = config.get('output_dir', 'results/lm_run')
        os.makedirs(self.output_dir, exist_ok=True)
        logger.info(f"Trainer initialized. Device: {self.device}, AMP: {self.use_amp}, Output Dir: {self.output_dir}")

        self.logger = logging.getLogger(__name__)

        self.best_val_metric = float('inf') # Initialize for perplexity (lower is better)
        self.epochs_no_improve = 0

    def _train_epoch(self, train_loader):
        """Runs a single training epoch."""
        self.generator.train()
        total_loss = 0
        progress_bar = tqdm(train_loader, desc="Training Epoch", leave=False)

        for i, batch in enumerate(progress_bar):
            if i % 100 == 0:
                print(f"  Processed training batch {i}") # Added for debugging progress
            self.optimizer.zero_grad()

            # Move batch to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)

            # Use Automatic Mixed Precision (AMP) if enabled
            with torch.amp.autocast(device_type=self.device.type, enabled=self.use_amp):
                outputs = self.generator(input_ids, attention_mask=attention_mask)
                logits = outputs # Assuming generator directly outputs logits

                # Calculate loss
                # CrossEntropyLoss expects logits as (N, C, ...) and labels as (N, ...)
                # N = batch_size * seq_len, C = vocab_size
                loss = self.criterion(logits.view(-1, logits.size(-1)), labels.view(-1))

            # Backward pass and optimization
            if self.scaler: # AMP enabled
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else: # AMP disabled
                loss.backward()
                self.optimizer.step()

            total_loss += loss.item()
            progress_bar.set_postfix({'loss': f"{loss.item():.4f}"}) 

        avg_train_loss = total_loss / len(train_loader)
        self.logger.info(f"Epoch Train Loss: {avg_train_loss:.4f}")
        return avg_train_loss

    def _validate_epoch(self, val_loader):
        """Runs a single validation epoch."""
        self.generator.eval()
        total_loss = 0
        total_batches = 0
        progress_bar = tqdm(val_loader, desc="Validation Epoch", leave=False)

        with torch.no_grad():
            for i, batch in enumerate(progress_bar):
                # Move batch to device
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                with torch.amp.autocast(device_type=self.device.type, enabled=self.use_amp):
                    outputs = self.generator(input_ids, attention_mask=attention_mask)
                    logits = outputs
                    loss = self.criterion(logits.view(-1, logits.size(-1)), labels.view(-1))

                # Only calculate loss and add if not empty (avoid NaN from potential empty batches)
                if labels.numel() > 0:
                    total_loss += loss.item() * input_ids.size(0) # Weighted by batch size
                    total_batches += input_ids.size(0)
                    avg_batch_loss = loss.item()
                    batch_perplexity = torch.exp(loss).item() # Calculate perplexity for the batch
                    progress_bar.set_postfix({'val_loss': f"{avg_batch_loss:.4f}", 'val_ppl': f"{batch_perplexity:.2f}"})
                else:
                    progress_bar.set_postfix({'val_loss': 'N/A', 'val_ppl': 'N/A'})

        # Calculate average loss and perplexity across the epoch
        avg_val_loss = total_loss / total_batches if total_batches > 0 else float('inf')
        avg_perplexity = np.exp(avg_val_loss) if avg_val_loss != float('inf') else float('inf')

        self.logger.info(f"Epoch Validation Summary: Avg Loss = {avg_val_loss:.4f}, Avg Perplexity = {avg_perplexity:.2f}")
        return avg_perplexity # Return perplexity for early stopping check

    def train(self, train_loader, val_loader):
        """Main training loop for the language model."""
        if not train_loader or not val_loader:
            self.logger.error("Train or validation loader not provided to train method.")
            return

        logger.info(f"Starting training for {self.num_epochs} epochs...")

        for epoch in range(self.num_epochs):
            logger.info(f"--- Epoch {epoch+1}/{self.num_epochs} ---")

            # Training step
            avg_train_loss = self._train_epoch(train_loader)
            logger.info(f"Epoch {epoch+1} Training Summary: Avg Loss = {avg_train_loss:.4f}")

            # Validation step
            val_perplexity = self._validate_epoch(val_loader)

            # Early stopping based on validation perplexity (lower is better)
            if val_perplexity < self.best_val_metric:
                logger.info(f"Validation perplexity improved from {self.best_val_metric:.2f} to {val_perplexity:.2f}. Saving model...")
                self.best_val_metric = val_perplexity
                torch.save(self.generator.state_dict(), os.path.join(self.output_dir, 'best_generator_lm.pth'))
                # Save config for reference
                with open(os.path.join(self.output_dir, 'config.yaml'), 'w') as f:
                    yaml.dump(self.config, f)
                self.epochs_no_improve = 0
            else:
                self.epochs_no_improve += 1
                logger.info(f"Validation perplexity did not improve for {self.epochs_no_improve} epochs (Best: {self.best_val_metric:.2f}).")
                if self.epochs_no_improve >= self.patience:
                    logger.info(f"Early stopping triggered after {epoch+1} epochs.")
                    break

        logger.info("Training finished.")
        logger.info(f"Best validation perplexity achieved: {self.best_val_metric:.2f}")
        logger.info(f"Model saved to: {os.path.join(self.output_dir, 'best_generator_lm.pth')}")

    def save_checkpoint(self, filename='checkpoint.pth'):
        os.makedirs(self.output_dir, exist_ok=True)
        save_path = os.path.join(self.output_dir, filename)
        torch.save(self.generator.state_dict(), save_path)
        self.logger.info(f"Checkpoint saved to {save_path}")

    def evaluate(self): # Placeholder for test set evaluation
        self.logger.warning("Evaluation step needs implementation for Language Modeling.")
        # Load best model
        # Run on test set
        # Calculate perplexity
        pass
