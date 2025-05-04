import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
import torch.autograd as autograd
import torch.nn.functional as F

import logging
import os
import time
from datetime import datetime
from tqdm.auto import tqdm
import yaml
import evaluate # Hugging Face evaluate
from transformers import AutoTokenizer # Needed for Decoder
from transformers import get_cosine_schedule_with_warmup
import nltk
from nltk.util import ngrams
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction # Added for Self-BLEU
import ssl
import random # <<< Add this import
from torch.utils.tensorboard import SummaryWriter # Import SummaryWriter
import numpy as np  # Added for BERTScore averaging

# Ensure NLTK data is available (might be better in main.py or setup script)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Import new model components
from model.text_encoder import SentenceEncoder
from model.gan_networks import LatentGenerator, LatentDiscriminator
from model.text_decoder import TextDecoder

# Import data loader function
from data.dataset import load_and_prepare_dataset

logger = logging.getLogger(__name__)

# Helper function to set device
def get_device(device_config: str) -> torch.device:
    if device_config == 'auto':
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_config)

def calculate_distinct_n(sentences, n):
    """Calculates distinct-n score for a list of sentences."""
    if not sentences:
        return 0.0
    
    all_ngrams = []
    total_ngrams_count = 0
    
    for sentence in sentences:
        tokens = nltk.word_tokenize(sentence.lower()) # Tokenize and lowercase
        current_ngrams = list(ngrams(tokens, n))
        all_ngrams.extend(current_ngrams)
        total_ngrams_count += len(current_ngrams)
        
    if total_ngrams_count == 0:
        return 0.0
        
    distinct_ngrams_count = len(set(all_ngrams))
    return distinct_ngrams_count / total_ngrams_count

class Trainer:
    """Trains a Latent Space GAN using WGAN-GP."""
    def __init__(self, config_path: str, resume_checkpoint: str = None):
        logger.info(f"Initializing Trainer with config: {config_path}")
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # Store resume path if provided
        self.resume_checkpoint = resume_checkpoint

        # --- Setup --- 
        self.seed = self.config.get('seed', 42)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

        self.output_dir = self.config.get('output_dir', 'results/latent_gan_run')
        os.makedirs(self.output_dir, exist_ok=True)

        # --- WGAN-GP Specific Config ---
        self.gradient_penalty_lambda = self.config['training'].get('gradient_penalty_lambda', 10.0)
        self.critic_iterations = self.config['training'].get('critic_iterations', 5)
        logger.info(f"WGAN-GP Config: Lambda={self.gradient_penalty_lambda}, Critic Iterations={self.critic_iterations}")

        self.device = get_device(self.config.get('training', {}).get('device', 'auto'))
        logger.info(f"Using device: {self.device}")

        self.use_amp = self.config['training'].get('use_amp', False) and torch.cuda.is_available()
        self.scaler = GradScaler('cuda', enabled=self.use_amp) # New API
        logger.info(f"Using Mixed Precision (AMP): {self.use_amp}")

        # --- Load Tokenizer (for Decoder) --- 
        # Use a standard tokenizer, e.g., based on the encoder or a default one
        # Needs vocab_size consistency with decoder config
        tokenizer_name = self.config['model']['encoder'].get('name', 'bert-base-uncased') 
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
                # Resize embedding layer of decoder later if necessary
            self.pad_token_id = self.tokenizer.pad_token_id
            self.vocab_size = self.tokenizer.vocab_size
            logger.info(f"Loaded Tokenizer: {tokenizer_name} with Vocab Size: {self.vocab_size}, PAD ID: {self.pad_token_id}")
            # Update config's vocab_size if needed (or ensure consistency)
            if self.config['model']['decoder']['vocab_size'] != self.vocab_size:
                 logger.warning(f"Config vocab_size ({self.config['model']['decoder']['vocab_size']}) differs from tokenizer ({self.vocab_size}). Using tokenizer's.")
                 self.config['model']['decoder']['vocab_size'] = self.vocab_size
        except Exception as e:
            logger.error(f"Failed to load tokenizer '{tokenizer_name}': {e}. Check name/connection.", exc_info=True)
            raise

        # --- Data --- 
        logger.info("Loading dataset...")
        # Pass only the data part of the config
        self.train_loader, self.val_loader, self.test_loader, _vocab_size_from_data = load_and_prepare_dataset(self.config['data'])
        logger.info("Dataset loaded.")
        if self.test_loader:
            logger.info(f"Test dataloader loaded with {len(self.test_loader)} batches.")
        else:
            logger.warning("Test dataloader was not created/returned.")

        # --- Models --- 
        logger.info("Initializing models...")
        model_config = self.config['model']
        train_config = self.config['training']
        
        # 1. Encoder
        self.encoder = SentenceEncoder(
            model_name=model_config['encoder']['name'],
            device=self.device,
            freeze=model_config['encoder'].get('trainable', False) == False
        ).to(self.device)
        # Get latent_dim from encoder AFTER initialization
        self.latent_dim = self.encoder.latent_dim
        logger.info(f"Encoder initialized. Latent dim: {self.latent_dim}")

        # Update latent_dim in config for G/D if it wasn't set or differs
        if model_config['gan_generator'].get('latent_dim') != self.latent_dim:
             logger.warning("Updating GAN Generator latent_dim to match Encoder.")
             model_config['gan_generator']['latent_dim'] = self.latent_dim
        if model_config['gan_discriminator'].get('latent_dim') != self.latent_dim:
             logger.warning("Updating GAN Discriminator latent_dim to match Encoder.")
             model_config['gan_discriminator']['latent_dim'] = self.latent_dim
        if model_config['decoder'].get('latent_dim') != self.latent_dim:
             logger.warning("Updating Decoder latent_dim to match Encoder.")
             model_config['decoder']['latent_dim'] = self.latent_dim
             
        # 2. GAN Generator
        gen_config = model_config['gan_generator']
        self.generator = LatentGenerator(
            noise_dim=gen_config['noise_dim'],
            latent_dim=self.latent_dim,
            hidden_dims=gen_config['hidden_dims']
        ).to(self.device)
        logger.info("GAN Generator initialized.")

        # 3. GAN Discriminator
        disc_config = model_config['gan_discriminator']
        self.discriminator = LatentDiscriminator(
            latent_dim=self.latent_dim,
            hidden_dims=disc_config['hidden_dims']
        ).to(self.device)
        logger.info("GAN Discriminator initialized.")

        # 4. Decoder (optional for training if no reconstruction loss, but needed for eval)
        dec_config = model_config['decoder']
        self.decoder = TextDecoder(
            latent_dim=self.latent_dim,
            vocab_size=self.vocab_size, # Use loaded tokenizer vocab size
            d_model=dec_config['d_model'],
            n_head=dec_config['n_head'],
            n_layer=dec_config['n_layer'],
            d_ff=dec_config['d_ff'],
            max_length=dec_config['max_length'],
            dropout=dec_config.get('dropout', 0.1),
            pad_token_id=self.pad_token_id 
        ).to(self.device)
        # Resize embeddings if tokenizer added pad token
        self.decoder.token_embedding.weight.requires_grad = False # Freeze embedding temporarily
        self.decoder.token_embedding = nn.Embedding.from_pretrained(self.decoder.token_embedding.weight[:self.tokenizer.vocab_size-1,:]) # remove last row
        self.decoder.token_embedding = nn.Embedding(self.tokenizer.vocab_size, dec_config['d_model'], padding_idx=self.pad_token_id).to(self.device)
        # self.decoder.token_embedding.weight.data[-1] = 0 # Initialize pad embedding maybe? 
        self.decoder.output_projection = nn.Linear(dec_config['d_model'], self.vocab_size).to(self.device) # Resize output layer too
        self.decoder.vocab_size = self.vocab_size # Update internal vocab size
        self.decoder.token_embedding.weight.requires_grad = True # Unfreeze
        logger.info("Text Decoder initialized and embeddings potentially resized.")

        # --- Optimizers --- 
        logger.info("Initializing optimizers...")
        optimizer_config = self.config['training']
        lr_g = float(optimizer_config['lr_g']) # Cast to float
        lr_d = float(optimizer_config['lr_d']) # Cast to float
        # WGAN-GP recommended betas
        betas_g = tuple(optimizer_config.get('betas_g', [0.5, 0.999]))
        betas_d = tuple(optimizer_config.get('betas_d', [0.5, 0.999]))

        self.optimizer_g = optim.Adam(self.generator.parameters(), lr=lr_g, betas=betas_g)
        self.optimizer_d = optim.Adam(self.discriminator.parameters(), lr=lr_d, betas=betas_d)
        logger.info(f"Optimizers initialized: G (lr={lr_g}, betas={betas_g}), D (lr={lr_d}, betas={betas_d})")
        
        # Autoencoder optimizer and reconstruction criterion
        lr_ae = float(optimizer_config.get('lr_ae', 0.0))
        betas_ae = tuple(optimizer_config.get('betas_ae', [0.9, 0.999]))
        self.criterion_recon = nn.CrossEntropyLoss(ignore_index=self.pad_token_id)
        self.optimizer_ae = optim.Adam(
            list(filter(lambda p: p.requires_grad, self.encoder.parameters())) + list(self.decoder.parameters()),
            lr=lr_ae,
            betas=betas_ae
        )
        logger.info(f"Optimizer AE initialized: lr={lr_ae}, betas={betas_ae}")
        # --- Learning Rate Schedulers ---
        num_epochs = self.config['training'].get('num_epochs', 10)
        num_training_steps = num_epochs * len(self.train_loader)
        warmup_ratio = optimizer_config.get('warmup_ratio', 0.1)
        num_warmup_steps = int(warmup_ratio * num_training_steps)
        self.scheduler_g = get_cosine_schedule_with_warmup(self.optimizer_g, num_warmup_steps, num_training_steps)
        self.scheduler_d = get_cosine_schedule_with_warmup(self.optimizer_d, num_warmup_steps, num_training_steps)
        self.scheduler_ae = get_cosine_schedule_with_warmup(self.optimizer_ae, num_warmup_steps, num_training_steps) if lr_ae > 0 else None
        logger.info(f"Schedulers initialized: warmup_steps={num_warmup_steps}, total_steps={num_training_steps}")

        # --- Training State --- 
        self.start_epoch = 0
        self.global_step = 0
        # Store epoch losses for summary
        self.last_epoch_losses = {'D_Loss': 0.0, 'G_Loss': 0.0, 'GP': 0.0}

        # --- Load Checkpoint (Optional) ---
        # checkpoint_path = self.config.get('load_checkpoint')

        # --- Dummy op to initialize CUDA context --- (Fixes cuBLAS warning)
        if self.device.type == 'cuda':
            try:
                _ = torch.tensor([1.0], device=self.device) + torch.tensor([1.0], device=self.device)
                logger.debug("Dummy CUDA operation successful.")
            except Exception as e:
                logger.warning(f"Dummy CUDA operation failed: {e}")

        # --- WGAN-GP Params --- 
        self.n_critic = train_config.get('n_critic', 5)
        self.lambda_gp = train_config.get('lambda_gp', 10)
        self.noise_dim = gen_config['noise_dim']

        # --- TensorBoard Writer --- 
        run_name = self.config.get('run_name', 'latent_gan_run')
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = os.path.join(self.output_dir, f"{run_name}_{timestamp}")
        os.makedirs(self.run_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=self.run_dir)
        logger.info(f"TensorBoard logs will be saved to: {self.run_dir}")

        self.current_epoch = 0
        self.global_step = 0 # Initialize global step counter for TensorBoard
        self.decoder_max_length = self.config['model']['decoder']['max_length'] # Store max length

        # --- Load Checkpoint (Optional) ---
        # checkpoint_path = self.config.get('load_checkpoint')

        self.best_metric_value = -1.0 # Initialize best metric score (assuming higher is better, e.g., BLEU)
        self.best_metric_name = self.config.get('evaluation', {}).get('best_metric', 'BLEU-4')
        logger.info(f"Tracking best model based on validation metric: {self.best_metric_name}")

    def _compute_gradient_penalty(self, real_samples: torch.Tensor, fake_samples: torch.Tensor) -> torch.Tensor:
        """Calculates the gradient penalty loss for WGAN GP"""
        # Random weight term for interpolation between real and fake samples
        alpha = torch.rand(real_samples.size(0), 1, device=self.device)
        alpha = alpha.expand(real_samples.size())

        # Get random interpolation between real and fake samples
        interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)

        with autocast(self.device.type, enabled=self.use_amp):
            d_interpolates = self.discriminator(interpolates)

        # Get gradient w.r.t. interpolates
        gradients = autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=torch.ones(d_interpolates.size(), device=self.device),
            create_graph=True, # Create graph for second derivative calculation
            retain_graph=True, # Retain graph for G pass
            only_inputs=True,
        )[0]

        gradients = gradients.view(gradients.size(0), -1)
        # Calculate the gradient penalty: (||grad(D(interpolates))||_2 - 1)^2
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty * self.gradient_penalty_lambda

    def _train_epoch(self, epoch: int):
        """Runs a single training epoch for the WGAN-GP."""
        self.generator.train()
        self.discriminator.train()
        self.encoder.train() # Keep encoder in train mode if used

        epoch_start_time = time.time()
        d_losses, g_losses, gp_losses = [], [], []
        total_batches = len(self.train_loader)
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch+1} Training", leave=False, unit='batch')

        for i, batch in enumerate(progress_bar):
            # Move data to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)

            # --------------------- #
            #  Train Discriminator
            # --------------------- #
            self.optimizer_d.zero_grad()

            with autocast(self.device.type, enabled=self.use_amp):
                # Encode real sentences to get real latent vectors
                with torch.no_grad(): # Don't need gradients for encoder when training D
                    real_latents = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
                    if isinstance(real_latents, tuple): # Handle potential tuple output from encoder
                        real_latents = real_latents[0]
                real_latents = real_latents.detach() # Detach from encoder graph

                # Generate fake latent vectors from noise
                noise = torch.randn(input_ids.size(0), self.noise_dim, device=self.device)
                # Detach fake latents so generator isn't updated here
                fake_latents = self.generator(noise).detach()

                # Get discriminator scores for real and fake latents
                real_scores = self.discriminator(real_latents)
                fake_scores = self.discriminator(fake_latents)

                # Calculate gradient penalty
                gradient_penalty = self._compute_gradient_penalty(real_latents, fake_latents)

                # Calculate WGAN-GP discriminator loss
                # Loss_D = E[D(fake)] - E[D(real)] + lambda * GP
                d_loss = torch.mean(fake_scores) - torch.mean(real_scores) + gradient_penalty

            # Scale loss, perform backward pass, and update discriminator weights
            self.scaler.scale(d_loss).backward()
            self.scaler.step(self.optimizer_d)
            self.scaler.update() # Updates the scale for next iteration.

            # Store losses for logging
            d_losses.append(d_loss.item())
            gp_losses.append(gradient_penalty.item())

            # ----------------- #
            #  Train Generator
            # ----------------- #
            # Train the generator only every `critic_iterations` steps
            if (i + 1) % self.critic_iterations == 0:
                self.optimizer_g.zero_grad()

                with autocast(self.device.type, enabled=self.use_amp):
                    # Generate fake latents (requires grad this time)
                    noise = torch.randn(input_ids.size(0), self.noise_dim, device=self.device)
                    fake_latents_g = self.generator(noise)

                    # Get discriminator scores for fake latents
                    fake_scores_g = self.discriminator(fake_latents_g)

                    # Calculate WGAN generator loss
                    # Loss_G = -E[D(fake)] (Maximize the critic's score for fake samples)
                    g_loss = -torch.mean(fake_scores_g)

                # Scale loss, perform backward pass, and update generator weights
                self.scaler.scale(g_loss).backward()
                self.scaler.step(self.optimizer_g)
                self.scaler.update()

                # Store generator loss
                g_losses.append(g_loss.item())

                # Step the generator's scheduler only when the generator is updated
                if self.scheduler_g:
                    self.scheduler_g.step()

            # Step the discriminator's scheduler every step
            if self.scheduler_d:
                self.scheduler_d.step()

            # --- Logging & Progress Bar --- 
            self.global_step += 1
            current_lr_g = self.optimizer_g.param_groups[0]['lr']
            current_lr_d = self.optimizer_d.param_groups[0]['lr']
            
            log_dict = {
                'D_Loss': f"{d_loss.item():.4f}",
                'GP': f"{gradient_penalty.item():.4f}",
                'LR_D': f"{current_lr_d:.2e}",
                'LR_G': f"{current_lr_g:.2e}"
            }
            # Only add G_Loss to log if it was calculated in this step
            if (i + 1) % self.critic_iterations == 0 and g_losses:
                log_dict['G_Loss'] = f"{g_losses[-1]:.4f}"
            else: # Keep placeholder if G wasn't trained or list is empty
                log_dict['G_Loss'] = "N/A"
                
            progress_bar.set_postfix(log_dict)
            
            # Log to TensorBoard periodically
            if self.writer and self.global_step % self.config['training'].get('log_interval', 50) == 0:
                self.writer.add_scalar('Loss/Discriminator', d_loss.item(), self.global_step)
                self.writer.add_scalar('Loss/Gradient_Penalty', gradient_penalty.item(), self.global_step)
                self.writer.add_scalar('LR/Discriminator', current_lr_d, self.global_step)
                self.writer.add_scalar('LR/Generator', current_lr_g, self.global_step)
                # Log G loss only when it's computed
                if (i + 1) % self.critic_iterations == 0 and g_losses:
                    self.writer.add_scalar('Loss/Generator', g_losses[-1], self.global_step)
                    
        # --- End of Epoch Summary --- 
        epoch_end_time = time.time()
        avg_d_loss = sum(d_losses) / len(d_losses) if d_losses else 0
        avg_g_loss = sum(g_losses) / len(g_losses) if g_losses else 0 # Avg only over steps where G was trained
        avg_gp = sum(gp_losses) / len(gp_losses) if gp_losses else 0

        logger.info(f"Epoch {epoch+1} Summary | Time: {epoch_end_time - epoch_start_time:.2f}s | Avg D Loss: {avg_d_loss:.4f} | Avg G Loss: {avg_g_loss:.4f} | Avg GP: {avg_gp:.4f}")
        
        # Store last epoch losses for checkpoint saving
        self.last_epoch_losses = {'D_Loss': avg_d_loss, 'G_Loss': avg_g_loss, 'GP': avg_gp}

        # Log epoch averages to TensorBoard
        if self.writer:
            self.writer.add_scalar('Epoch_Loss/Discriminator', avg_d_loss, epoch + 1)
            self.writer.add_scalar('Epoch_Loss/Generator', avg_g_loss, epoch + 1)
            self.writer.add_scalar('Epoch_Loss/Gradient_Penalty', avg_gp, epoch + 1)

    def _evaluate(self, epoch: int):
        """Evaluate the model on the validation set, compute metrics, and track the best model."""
        if not self.val_loader:
            logger.warning("Validation loader not available, skipping evaluation.")
            return

        self.generator.eval()
        self.discriminator.eval()
        self.encoder.eval()
        self.decoder.eval()

        all_generated_texts = []
        all_reference_texts = [] # List of lists for BLEU/METEOR/ROUGE

        # --- Get References from Validation Set ---
        num_samples_to_generate = 0
        try:
            # Get a batch of references
            ref_batch = next(iter(self.val_loader))
            # Assuming ref_batch['text'] is a list of strings -> Correction: Assume ref_batch IS the list
            # references = ref_batch['text'] # Error occurs here if ref_batch is a list
            if isinstance(ref_batch, list):
                references = ref_batch
            elif isinstance(ref_batch, dict) and 'text' in ref_batch:
                 # Keep handling dict just in case dataloader changes
                references = ref_batch['text']
            else:
                 logger.warning(f"Unexpected validation batch format: {type(ref_batch)}. Cannot extract references.")
                 references = []

            num_samples_to_generate = len(references)
            # Format for evaluate library (list of lists)
            all_reference_texts = [[ref] for ref in references]
            logger.info(f"Using {num_samples_to_generate} reference texts from validation set.")
        except StopIteration:
            logger.warning("Validation loader is empty. Cannot get references.")
            return
        except Exception as e:
            logger.error(f"Error fetching references from validation loader: {e}", exc_info=True)
            return

        # --- Generate Samples ---
        if num_samples_to_generate > 0:
            logger.info(f"Generating {num_samples_to_generate} samples...")
            try:
                all_generated_texts = self.generate_text_from_noise(
                    num_samples=num_samples_to_generate,
                    max_length=self.decoder_max_length
                )
                logger.info(f"Generated {len(all_generated_texts)} samples.")
                if all_generated_texts:
                    logger.info(f"Example Generated Text: {random.choice(all_generated_texts)}")
                else:
                     logger.warning("Text generation returned an empty list.")
            except Exception as e:
                logger.error(f"Error during text generation for evaluation: {e}", exc_info=True)
                all_generated_texts = [] # Ensure it's an empty list on error

        # --- Calculate Metrics ---
        metrics_summary = { # Initialize summary dictionary
            'epoch': epoch + 1,
            **self.last_epoch_losses # Add losses from training
        }

        if not all_generated_texts:
            logger.warning("Skipping all metric calculations as text generation failed or was skipped.")
        else:
            # --- Diversity Metrics (Distinct-N & Self-BLEU) ---
            logger.info("Computing Diversity metrics (Distinct-N, Self-BLEU)...")
            try:
                distinct_1 = calculate_distinct_n(all_generated_texts, 1)
                distinct_2 = calculate_distinct_n(all_generated_texts, 2)
                self_bleu4 = calculate_self_bleu(all_generated_texts, n=4)
                metrics_summary['Distinct-1'] = distinct_1
                metrics_summary['Distinct-2'] = distinct_2
                metrics_summary['Self-BLEU4'] = self_bleu4
                logger.info(f"Evaluation Metrics - Distinct-1: {distinct_1:.4f}, Distinct-2: {distinct_2:.4f}, Self-BLEU4: {self_bleu4:.4f}")
                if self.writer:
                    self.writer.add_scalar('Evaluation/Distinct-1', distinct_1, epoch + 1)
                    self.writer.add_scalar('Evaluation/Distinct-2', distinct_2, epoch + 1)
                    self.writer.add_scalar('Evaluation/Self-BLEU4', self_bleu4, epoch + 1)
            except Exception as e:
                logger.error(f"Error calculating diversity metrics: {e}", exc_info=True)

            # --- Reference-Based Metrics (BLEU, METEOR, ROUGE) ---
            if all_reference_texts:
                logger.info("Computing BLEU, METEOR, ROUGE vs validation set...")
                try:
                    bleu_metric = evaluate.load('bleu')
                    meteor_metric = evaluate.load('meteor')
                    rouge_metric = evaluate.load('rouge')

                    # BLEU Computation (get full results)
                    bleu_results = bleu_metric.compute(predictions=all_generated_texts, references=all_reference_texts)
                    bleu_score = bleu_results['bleu'] # Overall BLEU-4
                    bleu_precisions = bleu_results.get('precisions', [0.0] * 4) # Extract BLEU-1 to 4
                    metrics_summary['BLEU-1'] = bleu_precisions[0]
                    metrics_summary['BLEU-2'] = bleu_precisions[1]
                    metrics_summary['BLEU-3'] = bleu_precisions[2]
                    metrics_summary['BLEU-4'] = bleu_precisions[3] # Same as 'bleu'
                    logger.info(f"Evaluation Metrics - BLEU-1: {bleu_precisions[0]:.4f}, BLEU-2: {bleu_precisions[1]:.4f}, BLEU-3: {bleu_precisions[2]:.4f}, BLEU-4: {bleu_precisions[3]:.4f}")
                    if self.writer:
                         self.writer.add_scalar('Evaluation/BLEU-1', bleu_precisions[0], epoch + 1)
                         self.writer.add_scalar('Evaluation/BLEU-2', bleu_precisions[1], epoch + 1)
                         self.writer.add_scalar('Evaluation/BLEU-3', bleu_precisions[2], epoch + 1)
                         self.writer.add_scalar('Evaluation/BLEU-4', bleu_precisions[3], epoch + 1)

                    # METEOR Computation
                    meteor_results = meteor_metric.compute(predictions=all_generated_texts, references=all_reference_texts)
                    meteor_score = meteor_results['meteor']
                    metrics_summary['METEOR'] = meteor_score
                    logger.info(f"Evaluation Metrics - METEOR: {meteor_score:.4f}")
                    if self.writer:
                        self.writer.add_scalar('Evaluation/METEOR', meteor_score, epoch + 1)

                    # ROUGE Computation
                    rouge_results = rouge_metric.compute(predictions=all_generated_texts, references=all_reference_texts)
                    rouge_l_score = rouge_results['rougeL'] # Use ROUGE-L
                    metrics_summary['ROUGE-L'] = rouge_l_score
                    logger.info(f"Evaluation Metrics - ROUGE-L: {rouge_l_score:.4f}")
                    if self.writer:
                        self.writer.add_scalar('Evaluation/ROUGE-L', rouge_l_score, epoch + 1)
                except Exception as e:
                    logger.error(f"Error computing reference-based metrics during evaluation: {e}", exc_info=True)
                    # Provide default values if calculation fails
                    metrics_summary.update({'BLEU-1': 0.0, 'BLEU-2': 0.0, 'BLEU-3': 0.0, 'BLEU-4': 0.0, 'METEOR': 0.0, 'ROUGE-L': 0.0})
            else:
                logger.warning("Skipping reference-based metrics as no references were available.")

            # --- BERTScore ---
            try:
                logger.info("Attempting to load BERTScore model...") 
                # Load BERTScore with roberta-base to reduce memory
                bertscore = evaluate.load('bertscore', model_type='roberta-base')
                logger.info(f"BERTScore model loaded: {bertscore.model_type if hasattr(bertscore, 'model_type') else 'N/A'}") 
                
                logger.info("Attempting to compute BERTScore...") 
                bert_res = bertscore.compute(predictions=all_generated_texts,
                                            references=[r[0] for r in all_reference_texts],
                                            lang='en')
                logger.info("BERTScore computation finished.") 
                metrics_summary['BERTScore_P'] = float(np.mean(bert_res['precision']))
                metrics_summary['BERTScore_R'] = float(np.mean(bert_res['recall']))
                metrics_summary['BERTScore_F1'] = float(np.mean(bert_res['f1']))
                logger.info(f"Evaluation Metrics - BERTScore F1: {metrics_summary['BERTScore_F1']:.4f}")
                if self.writer:
                    self.writer.add_scalar('Evaluation/BERTScore_F1', metrics_summary['BERTScore_F1'], epoch + 1)
            except Exception as e:
                logger.error(f"Error computing BERTScore: {e}")

        # --- Check if this is the best model so far based on the chosen metric --- 
        current_metric_value = metrics_summary.get(self.best_metric_name, -1.0)
        is_best = False
        # Only consider best model during GAN training phase (after pretraining)
        if epoch >= self.config['training'].get('pretrain_ae_epochs', 0):
            if current_metric_value > self.best_metric_value:
                self.best_metric_value = current_metric_value
                is_best = True
                logger.info(f"*** New best model found at epoch {epoch+1} with {self.best_metric_name}: {current_metric_value:.4f} ***")
        
        # --- Save Checkpoint (including best model if applicable) ---
        # We save the regular checkpoint regardless, but pass 'is_best' flag
        self._save_checkpoint(epoch, is_best=is_best) 
        # Note: _save_checkpoint is now called here instead of at the end of train() loop for the evaluated epoch

        # Log detailed metrics to console/file
        metrics_str = ", ".join([f"{k}: {v:.4f}" for k, v in metrics_summary.items()])
        logger.info(f"Evaluation Epoch {epoch+1} Metrics - {metrics_str}")

    def _save_checkpoint(self, epoch: int, is_best: bool = False):
        """Saves the current state of the models and optimizers."""
        if not self.output_dir:
            logger.warning("Checkpoint directory not specified. Skipping saving.")
            return

        checkpoint_state = {
            'epoch': epoch,
            'global_step': self.global_step,
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'encoder_state_dict': self.encoder.state_dict(), # Save even if frozen
            'decoder_state_dict': self.decoder.state_dict(),
            'optimizer_g_state_dict': self.optimizer_g.state_dict(),
            'optimizer_d_state_dict': self.optimizer_d.state_dict(),
            'optimizer_ae_state_dict': self.optimizer_ae.state_dict(),
            'scaler_state_dict': self.scaler.state_dict(),
            'config': self.config,
            'vocab_size': self.vocab_size,
            'pad_token_id': self.pad_token_id,
            'best_metric_value': self.best_metric_value, # Save best metric score
            'best_metric_name': self.best_metric_name, # Save which metric was used
        }
        latest_filename = os.path.join(self.output_dir, f"checkpoint_epoch_{epoch+1}.pth.tar")
        torch.save(checkpoint_state, latest_filename)
        logger.info(f"Checkpoint saved to {latest_filename}")

        # If this is the best model so far, save it to a fixed 'best_model' file
        if is_best:
            best_filename = os.path.join(self.output_dir, "best_model.pth.tar")
            torch.save(checkpoint_state, best_filename) # Overwrites previous best
            # Optionally use shutil.copyfile(latest_filename, best_filename) if state dict is huge
            logger.info(f"*** Best model checkpoint saved to {best_filename} (Metric: {self.best_metric_name}={self.best_metric_value:.4f}) ***")

    def load_checkpoint(self, checkpoint_path=None, load_best=False):
        """Loads a checkpoint to resume training. Can load the latest, a specific path, or the best model."""
        load_path = None
        if load_best:
            load_path = os.path.join(self.output_dir, "best_model.pth.tar")
            if not os.path.exists(load_path):
                 logger.warning(f"'load_best' is True, but best model file not found at {load_path}. Trying latest.")
                 load_path = None # Fallback to finding latest
        elif checkpoint_path: # Specific path provided
            load_path = checkpoint_path
        
        # If no specific path or best model requested/found, find the latest checkpoint
        if not load_path and self.output_dir and os.path.exists(self.output_dir):
             # Find the latest checkpoint file if path not provided
             checkpoint_files = glob.glob(os.path.join(self.output_dir, 'checkpoint_epoch_*.pth.tar'))
             if not checkpoint_files:
                 logger.warning("No checkpoint files found in the specified directory.")
                 return
             else:
                 # Sort by epoch number (extracted from filename)
                 latest_checkpoint = max(checkpoint_files, key=lambda p: int(re.search(r'_epoch_(\d+)\.pth\.tar$', p).group(1)))
                 load_path = latest_checkpoint
 
        if not load_path or not os.path.exists(load_path):
            logger.warning("No checkpoint file specified or found. Starting from scratch.")
            return
 
        logger.info(f"Loading checkpoint from {load_path}...")
        checkpoint = torch.load(load_path, map_location=self.device)
 
        # Load model states
        self.generator.load_state_dict(checkpoint['generator_state_dict'])
        self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        self.encoder.load_state_dict(checkpoint['encoder_state_dict'])
        self.decoder.load_state_dict(checkpoint['decoder_state_dict'])
 
        # Load optimizer states
        self.optimizer_g.load_state_dict(checkpoint['optimizer_g_state_dict'])
        self.optimizer_d.load_state_dict(checkpoint['optimizer_d_state_dict'])
        self.optimizer_ae.load_state_dict(checkpoint['optimizer_ae_state_dict'])
 
        # Load scaler state
        self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
 
        # Load best metric tracking state if available
        self.best_metric_value = checkpoint.get('best_metric_value', -1.0)
        self.best_metric_name = checkpoint.get('best_metric_name', self.best_metric_name) # Keep config default if not in ckpt
 
        self.start_epoch = checkpoint['epoch'] + 1
        self.global_step = checkpoint.get('global_step', 0) # Added for compatibility
 
        logger.info(f"Checkpoint loaded. Resuming training from epoch {self.start_epoch}. Best {self.best_metric_name} so far: {self.best_metric_value:.4f}")
 
        # Check if loaded config matches current config (optional)
        if 'config' in checkpoint and checkpoint['config'] != self.config:
            logger.warning("Loaded checkpoint config does not match the current config. Continuing with the current config.")

    def train(self):
        """Main training loop across epochs."""
        logger.info("Starting training...")
        num_epochs = self.config['training'].get('num_epochs', 10)
        pretrain_epochs = self.config['training'].get('pretrain_ae_epochs', 0)
        eval_every = self.config.get('evaluation', {}).get('eval_every_epochs', 1)

        # --- Load Checkpoint --- 
        self.start_epoch = 0 # Default start epoch
        self.best_metric_value = -float('inf') if self.config['evaluation'].get('higher_is_better', True) else float('inf')

        if self.resume_checkpoint:
            logger.info(f"Attempting to resume from specified checkpoint: {self.resume_checkpoint}")
            self.load_checkpoint(checkpoint_path=self.resume_checkpoint)
        elif self.config['training'].get('resume', False):
            logger.info("Attempting to resume from latest checkpoint based on config...")
            self.load_checkpoint() # Will try latest or best based on load_checkpoint logic
        # ------------------------
        
        for epoch in range(self.start_epoch, num_epochs):
            epoch_start_time = time.time()
            # Warm-start MLE pretraining
            if epoch < pretrain_epochs:
                self._pretrain_ae_epoch(epoch)
            else:
                self._train_epoch(epoch)
            
            epoch_duration = time.time() - epoch_start_time
            logger.info(f"Epoch {epoch+1} completed in {epoch_duration:.2f} seconds.")

            if (epoch + 1) % eval_every == 0:
                self._evaluate(epoch)
            
            # Save checkpoint periodically for non-evaluation epochs
            # Checkpoints for evaluation epochs are saved within _evaluate() now
            elif (epoch + 1) % self.config['training'].get('save_every_epochs', 1) == 0: # Add save_every_epochs config
                self._save_checkpoint(epoch, is_best=False) # is_best is only determined during evaluation
 
        logger.info("Training finished.")
        self.test() # Call the new test method
        logger.info("Trainer.train() method finished.")

        # Close TensorBoard writer when training is done
        if self.writer:
            self.writer.close()

    def test(self):
        """Runs evaluation on the test dataset after training is complete."""
        if not self.test_loader:
            logger.warning("Test loader not available, skipping testing.")
            return

        logger.info("--- Starting Test Evaluation --- ")
        self.generator.eval()
        self.encoder.eval() # Encoder is needed to get latents from real text for comparison/metrics
        self.decoder.eval() # Decoder is needed to generate text from latents

        all_predictions = []
        all_references = [] # Using test set as references

        # Use the test loader
        pbar = tqdm(self.test_loader, desc="Testing")
        with torch.no_grad():
            for batch_idx, real_texts_batch in enumerate(pbar):
                # Use real texts from test set as references
                references_for_batch = [[text] for text in real_texts_batch] # Wrap each ref in a list for evaluate lib
                all_references.extend(references_for_batch)

                batch_size = len(real_texts_batch)
                
                # Generate fake samples from noise
                z = torch.randn(batch_size, self.noise_dim, device=self.device)
                fake_latents = self.generator(z)

                # Decode fake latents into text
                # Need to handle the decoder's input requirements
                # Assuming decoder needs a start token and generates sequentially
                # This part might need adjustment based on the specific TextDecoder implementation
                generated_texts = self.generate_text_from_latent(fake_latents)
                all_predictions.extend(generated_texts)

                # Optional: Log a few examples periodically
                if batch_idx % 100 == 0: 
                    logger.debug(f"Test Sample Prediction: {generated_texts[0]}")
                    logger.debug(f"Test Sample Reference: {references_for_batch[0][0]}")
        
        # --- Calculate Metrics --- 
        metrics = {}
        if all_predictions and all_references:
            # --- Create a random subset for expensive metrics ---
            num_total_samples = len(all_predictions)
            max_samples_subset = min(10000, num_total_samples) # Use 10k or total if less
            if num_total_samples > max_samples_subset:
                logger.info(f"Creating random subset of {max_samples_subset} samples for expensive metrics...")
                # Combine predictions and references to keep pairs aligned
                combined_samples = list(zip(all_predictions, all_references))
                random_subset_combined = random.sample(combined_samples, max_samples_subset)
                # Separate back into predictions and references
                predictions_subset = [item[0] for item in random_subset_combined]
                references_subset = [item[1] for item in random_subset_combined] # references_subset is list of lists
            else:
                 logger.info("Using all samples for expensive metrics as total is <= subset size.")
                 predictions_subset = all_predictions
                 # references_subset needs the same format as all_references (list of lists)
                 references_subset = all_references

            # --- BERTScore (on subset) ---
            try:
                logger.info("Attempting to load BERTScore model...") 
                bertscore = evaluate.load('bertscore', model_type='roberta-base')
                logger.info(f"BERTScore model loaded: {bertscore.model_type if hasattr(bertscore, 'model_type') else 'N/A'}") 
                
                logger.info(f"Attempting to compute BERTScore on {max_samples_subset} random samples...") 
                # Use the random subset
                bert_res = bertscore.compute(predictions=predictions_subset, 
                                            references=[r[0] for r in references_subset], # Extract inner list for bertscore 
                                            lang='en',
                                            batch_size=16)
                logger.info("BERTScore computation finished.") 
                metrics['bertscore_f1'] = float(np.mean(bert_res['f1']))
            except Exception as e:
                logger.error(f"Error computing BERTScore on test: {e}")
            
            # --- Reference-based metrics (BLEU, METEOR, ROUGE) on FULL set ---
            # These are usually less expensive, calculate on all samples
            try:
                logger.info("Computing BLEU, METEOR, ROUGE on FULL test set...") 
                bleu_result = evaluate.load('bleu').compute(predictions=all_predictions, references=all_references)
                metrics['bleu-4'] = bleu_result['bleu'] # Overall BLEU-4
                meteor_result = evaluate.load('meteor').compute(predictions=all_predictions, references=all_references)
                metrics['meteor'] = meteor_result['meteor']
                rouge_result = evaluate.load('rouge').compute(predictions=all_predictions, references=all_references)
                metrics['rougeL'] = rouge_result['rougeL']
            except Exception as e:
                logger.error(f"Error calculating non-BERTScore test metrics: {e}", exc_info=True) # Updated log message
                # Initialize all potentially calculated metrics to 0 on error
                metrics.update({'bleu-4': 0.0, 'meteor': 0.0, 'rougeL': 0.0})

            # --- Diversity Metrics (Distinct-N & Self-BLEU) ---
            try:
                logger.info("Computing Diversity metrics (Distinct-N, Self-BLEU)...") # ADDED
                metrics['distinct-1'] = calculate_distinct_n(predictions_subset, 1) # MODIFIED
                metrics['distinct-2'] = calculate_distinct_n(predictions_subset, 2) # MODIFIED
                metrics['self-bleu4'] = calculate_self_bleu(predictions_subset, n=4) # MODIFIED
                logger.info("Diversity metrics computation finished.") # ADDED
            except Exception as e:
                 logger.error(f"Error calculating diversity metrics on test: {e}", exc_info=True) # ADDED
                 metrics.update({'distinct-1': 0.0, 'distinct-2': 0.0, 'self-bleu4': 0.0}) # ADDED

        else:
            logger.warning("No predictions or references generated during testing, cannot compute metrics.")
            # Initialize all potentially calculated metrics to 0
            metrics = {'bertscore_f1': 0.0, 'bleu-4': 0.0, 'meteor': 0.0, 'rougeL': 0.0, 'distinct-1': 0.0, 'distinct-2': 0.0, 'self-bleu4': 0.0} # Updated init

        # --- Log Test Results --- 
        logger.info("--- Test Results --- ")
        # Define the order and desired names for logging
        # We use bleu-4 internally as it holds the overall score from evaluate.load('bleu')
        metric_log_order = {
            'bertscore_f1': 'BERTSCORE_F1',
            'bleu-4': 'BLEU',  # Log bleu-4 as BLEU
            'meteor': 'METEOR',
            'rougeL': 'ROUGEL',
            'distinct-1': 'DISTINCT_1',
            'distinct-2': 'DISTINCT_2',
            'self-bleu4': 'SELF_BLEU4'
        }
        
        for internal_name, log_name in metric_log_order.items():
            if internal_name in metrics:
                logger.info(f"Final Test {log_name}: {metrics[internal_name]:.4f}")
            else:
                 logger.warning(f"Metric '{internal_name}' not found in results for logging.") # Added warning
            
            # Optionally log to TensorBoard (e.g., using a large step number like self.total_epochs + 1)
            # self.writer.add_scalar(f'Test/{name}', score, self.config['training']['epochs'] + 1)
        logger.info("--- Test Evaluation Finished --- ")

    def generate_text_from_noise(self, num_samples=1, max_length=64):
        """Generates text samples from random noise using G and Decoder."""
        self.generator.eval()
        self.decoder.eval()
        
        noise = torch.randn(num_samples, self.noise_dim, device=self.device) 
        generated_texts = []

        try:
            with torch.no_grad():
                with autocast('cuda', enabled=self.use_amp): 
                    fake_latents = self.generator(noise)
                    
                    # Use decoder's generate method (assuming it exists and handles generation)
                    # Pass necessary generation parameters (e.g., start token, max length)
                    # We might need to get start_token_id similar to how it was done before
                    start_token_id = self.tokenizer.bos_token_id if self.tokenizer.bos_token_id is not None else self.tokenizer.cls_token_id
                    if start_token_id is None: 
                        start_token_id = self.tokenizer.pad_token_id
                        logger.warning(f"No BOS or CLS token found. Using PAD token ({start_token_id}) as start token for generation.")
                        
                    # Assuming decoder has a method like `generate` or similar
                    # The TextDecoder class might need its own generate method wrapping the underlying model's generate
                    generated_ids = self.decoder.generate(
                        latent_vector=fake_latents, 
                        max_new_tokens=max_length - 1, 
                        start_token_id=start_token_id 
                        # Add other generation params if needed (temperature, top_k etc.)
                        # These could be read from config['evaluation']['generation_params'] if desired
                    )
                    
            # Decode generated IDs to text
            generated_texts = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            
        except Exception as e:
            logger.error(f"Error during generate_text_from_noise: {e}", exc_info=True)
            # Return empty list or re-raise depending on desired handling
            return []

        return generated_texts

    def generate_text_from_latent(self, latents):
        """Generates text from the given latent vectors using the decoder."""
        self.decoder.eval()
        
        generated_texts = []

        try:
            with torch.no_grad():
                with autocast('cuda', enabled=self.use_amp): 
                    # Use decoder's generate method (assuming it exists and handles generation)
                    # Pass necessary generation parameters (e.g., start token, max length)
                    # We might need to get start_token_id similar to how it was done before
                    start_token_id = self.tokenizer.bos_token_id if self.tokenizer.bos_token_id is not None else self.tokenizer.cls_token_id
                    if start_token_id is None: 
                        start_token_id = self.tokenizer.pad_token_id
                        logger.warning(f"No BOS or CLS token found. Using PAD token ({start_token_id}) as start token for generation.")
                        
                    # Assuming decoder has a method like `generate` or similar
                    # The TextDecoder class might need its own generate method wrapping the underlying model's generate
                    generated_ids = self.decoder.generate(
                        latent_vector=latents, 
                        max_new_tokens=self.decoder_max_length - 1, 
                        start_token_id=start_token_id 
                        # Add other generation params if needed (temperature, top_k etc.)
                        # These could be read from config['evaluation']['generation_params'] if desired
                    )
                    
            # Decode generated IDs to text
            generated_texts = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            
        except Exception as e:
            logger.error(f"Error during generate_text_from_latent: {e}", exc_info=True)
            # Return empty list or re-raise depending on desired handling
            return []

        return generated_texts

# Helper function (outside class)
def calculate_distinct_n(sentences: list, n: int):
    """Calculates distinct-n score for a list of sentences."""
    if not sentences:
        return 0.0
    
    all_ngrams = []
    total_ngrams_count = 0
    
    for sentence in sentences:
        tokens = nltk.word_tokenize(sentence.lower()) 
        if not tokens: continue 
        current_ngrams = list(ngrams(tokens, n))
        all_ngrams.extend(current_ngrams)
        total_ngrams_count += len(current_ngrams)
        
    if total_ngrams_count == 0:
        return 0.0
        
    distinct_ngrams_count = len(set(all_ngrams))
    return distinct_ngrams_count / total_ngrams_count

def calculate_self_bleu(sentences: list, n: int = 4):
    """Calculates Self-BLEU score for a list of sentences."""
    if len(sentences) < 2:
        return 0.0 

    total_bleu = 0.0
    smoothing_function = SmoothingFunction().method1 
    weights = tuple(1. / n for _ in range(n)) 

    tokenized_sentences = [nltk.word_tokenize(s.lower()) for s in sentences]

    for i in range(len(tokenized_sentences)):
        hypothesis = tokenized_sentences[i]
        references = tokenized_sentences[:i] + tokenized_sentences[i+1:]
        if not hypothesis or not references: 
            continue
        # sentence_bleu expects references as a list of lists of tokens
        # Here, references is already a list of lists of tokens
        try:
            score = sentence_bleu(references, hypothesis, weights=weights, smoothing_function=smoothing_function)
            total_bleu += score
        except ZeroDivisionError: 
             pass 
        except Exception as e:
            logger.warning(f"Error during sentence_bleu calculation: {e}. Hyp: {hypothesis}, Ref sample: {references[0] if references else 'None'}")
            pass 

    # Average BLEU score over all sentences
    return total_bleu / len(tokenized_sentences) if tokenized_sentences else 0.0

# Example of how to run the trainer (usually called from main.py)
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Train Latent Space GAN')
    parser.add_argument('--config', type=str, required=True, help='Path to the configuration YAML file')
    parser.add_argument('--resume_checkpoint', type=str, default=None, help='Path to the checkpoint to resume from')
    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    trainer = Trainer(config_path=args.config, resume_checkpoint=args.resume_checkpoint)
    trainer.train()
