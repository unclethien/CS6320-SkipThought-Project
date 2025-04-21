import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
import torch.autograd as autograd

import logging
import os
import time
from datetime import datetime
from tqdm.auto import tqdm
import yaml
import evaluate # Hugging Face evaluate
from transformers import AutoTokenizer # Needed for Decoder
import nltk
from nltk.util import ngrams
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction # Added for Self-BLEU
import ssl
import random # <<< Add this import
from torch.utils.tensorboard import SummaryWriter # Import SummaryWriter

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
    def __init__(self, config_path: str):
        logger.info(f"Initializing Trainer with config: {config_path}")
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # --- Setup --- 
        self.seed = self.config.get('seed', 42)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

        self.output_dir = self.config.get('output_dir', 'results/latent_gan_run')
        os.makedirs(self.output_dir, exist_ok=True)

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
        self.train_loader, self.val_loader, _ = load_and_prepare_dataset(self.config['data'])
        logger.info("Dataset loaded.")

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
        betas_g = tuple(optimizer_config['betas_g'])
        betas_d = tuple(optimizer_config['betas_d'])

        self.optimizer_g = optim.Adam(self.generator.parameters(), lr=lr_g, betas=betas_g)
        self.optimizer_d = optim.Adam(self.discriminator.parameters(), lr=lr_d, betas=betas_d)
        logger.info(f"Optimizers initialized: G (lr={lr_g}, betas={betas_g}), D (lr={lr_d}, betas={betas_d})")
        
        # Optional: Optimizer for Autoencoder (Encoder + Decoder) if training end-to-end
        # self.optimizer_ae = ... 

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

    def _compute_gradient_penalty(self, real_samples: torch.Tensor, fake_samples: torch.Tensor) -> torch.Tensor:
        """Calculates the gradient penalty loss for WGAN GP"""
        # Random weight term for interpolation between real and fake samples
        alpha = torch.rand(real_samples.size(0), 1, device=self.device)
        alpha = alpha.expand(real_samples.size())
        
        # Get random interpolation between real and fake samples
        interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
        d_interpolates = self.discriminator(interpolates)
        
        # Get gradient w.r.t. interpolates
        gradients = autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=torch.ones_like(d_interpolates, requires_grad=False),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        
        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty

    def _train_epoch(self, epoch: int):
        """Runs a single training epoch for the WGAN-GP."""
        self.generator.train()
        self.discriminator.train()
        # Set encoder/decoder train/eval mode depending on if they are trained
        self.encoder.eval() # Assume frozen encoder for now
        self.decoder.eval() # Assume decoder is not trained here (only for eval)
        
        total_d_loss = 0.0
        total_g_loss = 0.0
        total_gp = 0.0
        batch_count = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}")
        for batch_idx, real_texts in enumerate(pbar):
            batch_size = len(real_texts)
            batch_count += 1
            
            # Ensure models are on the correct device (redundant if initialized correctly, but safe)
            self.discriminator.to(self.device)
            self.generator.to(self.device)
            self.encoder.to(self.device)

            # Convert real texts to real latent vectors
            with torch.no_grad(): # Encoder is frozen
                real_latents = self.encoder(real_texts).detach()
            real_latents = real_latents.to(self.device)

            # === Train Discriminator (Critic) ===
            self.optimizer_d.zero_grad()

            for _ in range(self.n_critic):
                # Generate fake latent vectors
                z = torch.randn(batch_size, self.noise_dim, device=self.device) # Use noise_dim
                with torch.no_grad(): # Don't track gradients for G when training D
                     fake_latents = self.generator(z).detach() 
                
                with torch.amp.autocast('cuda', enabled=self.use_amp): # New API
                    # Get scores for real and fake latents
                    real_scores = self.discriminator(real_latents)
                    fake_scores = self.discriminator(fake_latents)

                    # Calculate Gradient Penalty
                    gradient_penalty = self._compute_gradient_penalty(real_latents.data, fake_latents.data)
                    
                    # Calculate WGAN-GP Discriminator Loss
                    d_loss = fake_scores.mean() - real_scores.mean() + self.lambda_gp * gradient_penalty

                # Backward pass and optimization step for Discriminator
                self.scaler.scale(d_loss / self.n_critic).backward() # Accumulate gradients
                total_d_loss += d_loss.item() / self.n_critic
                total_gp += gradient_penalty.item() / self.n_critic

            self.scaler.step(self.optimizer_d)
            self.scaler.update()
            self.optimizer_d.zero_grad() # Zero grad again just in case

            # === Train Generator === 
            # Only update generator every n_critic steps (effectively)
            # But need to compute loss in each main loop iteration for logging?
            # Let's update G every main loop iteration (simpler, common practice)
            self.optimizer_g.zero_grad()

            # Generate new fake latents (need gradients now)
            z = torch.randn(batch_size, self.noise_dim, device=self.device) # Use noise_dim
            fake_latents_for_g = self.generator(z)
            
            with torch.amp.autocast('cuda', enabled=self.use_amp): # New API
                # Get discriminator score for fake latents
                fake_scores_for_g = self.discriminator(fake_latents_for_g)
                # Calculate Generator Loss (maximize D score for fakes -> minimize -D score)
                g_loss = -fake_scores_for_g.mean()
            
            # Backward pass and optimization step for Generator
            self.scaler.scale(g_loss).backward()
            self.scaler.step(self.optimizer_g)
            self.scaler.update()
            total_g_loss += g_loss.item()

            # --- Logging --- 
            self.global_step += 1
            if batch_idx % 50 == 0: # Log every 50 steps
                 pbar.set_postfix({
                     'D Loss': f'{d_loss.item():.4f}',
                     'G Loss': f'{g_loss.item():.4f}',
                     'GP': f'{gradient_penalty.item():.4f}'
                 })
            if self.writer:
                self.writer.add_scalar('train/D_loss_step', d_loss.item(), self.global_step)
                self.writer.add_scalar('train/G_loss_step', g_loss.item(), self.global_step)
                self.writer.add_scalar('train/GP_step', gradient_penalty.item(), self.global_step)
            # TODO: Add Tensorboard logging if desired

        avg_d_loss = total_d_loss / batch_count
        avg_g_loss = total_g_loss / batch_count
        avg_gp = total_gp / batch_count
        logger.info(f"Epoch {epoch+1} finished. Avg D Loss: {avg_d_loss:.4f}, Avg G Loss: {avg_g_loss:.4f}, Avg GP: {avg_gp:.4f}")
        if self.writer:
            self.writer.add_scalar('train/D_loss_epoch', avg_d_loss, epoch)
            self.writer.add_scalar('train/G_loss_epoch', avg_g_loss, epoch)
            self.writer.add_scalar('train/GP_epoch', avg_gp, epoch)

        # Store losses for summary
        self.last_epoch_losses = {'D_Loss': avg_d_loss, 'G_Loss': avg_g_loss, 'GP': avg_gp}

    def _evaluate(self, epoch: int):
        """Evaluate the model on the validation set and compute metrics."""
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
                    logger.error(f"Error computing reference-based metrics: {e}", exc_info=True)
            else:
                logger.warning("Skipping reference-based metrics as no references were available.")

        # --- Log Final Summary --- 
        logger.info(f"--- Evaluation Summary Epoch {epoch+1} ---")
        for key, value in metrics_summary.items():
            logger.info(f"{key}: {value:.4f}" if isinstance(value, float) else f"{key}: {value}")
        logger.info(f"----------------------------------------")

    def _save_checkpoint(self, epoch: int, is_best: bool = False):
        """Saves model and optimizer states."""
        checkpoint_state = {
            'epoch': epoch,
            'global_step': self.global_step,
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'encoder_state_dict': self.encoder.state_dict(), # Save even if frozen
            'decoder_state_dict': self.decoder.state_dict(),
            'optimizer_g_state_dict': self.optimizer_g.state_dict(),
            'optimizer_d_state_dict': self.optimizer_d.state_dict(),
            # Add optimizer_ae if used
            'scaler_state_dict': self.scaler.state_dict(),
            'config': self.config,
            'vocab_size': self.vocab_size,
            'pad_token_id': self.pad_token_id
        }
        filename = f'checkpoint_epoch_{epoch+1}.pth.tar'
        filepath = os.path.join(self.output_dir, filename)
        torch.save(checkpoint_state, filepath)
        logger.info(f"Checkpoint saved to {filepath}")
        
        if is_best:
             best_filepath = os.path.join(self.output_dir, 'checkpoint_best.pth.tar')
             torch.save(checkpoint_state, best_filepath)
             logger.info(f"Best checkpoint saved to {best_filepath}")

    # TODO: Implement checkpoint loading _load_checkpoint

    def train(self):
        """Main training loop across epochs."""
        logger.info("Starting training...")
        num_epochs = self.config['training'].get('num_epochs', 10)
        eval_every = self.config.get('evaluation', {}).get('eval_every_epochs', 1)
        
        for epoch in range(self.start_epoch, num_epochs):
            epoch_start_time = time.time()
            
            self._train_epoch(epoch)
            
            epoch_duration = time.time() - epoch_start_time
            logger.info(f"Epoch {epoch+1} completed in {epoch_duration:.2f} seconds.")

            if (epoch + 1) % eval_every == 0:
                self._evaluate(epoch)
            
            # Save checkpoint periodically (e.g., every epoch)
            self._save_checkpoint(epoch)

        logger.info("Training finished.")

        # Close TensorBoard writer when training is done
        if self.writer:
            self.writer.close()

    def generate_text_from_noise(self, num_samples=1, max_length=64):
        """Generates text samples from random noise using G and Decoder."""
        self.generator.eval()
        self.decoder.eval()
        
        noise = torch.randn(num_samples, self.noise_dim, device=self.device) # Use noise_dim
        generated_texts = []

        try:
            with torch.no_grad():
                with torch.amp.autocast('cuda', enabled=self.use_amp): # New API
                    fake_latents = self.generator(noise)
                    
                    # Use decoder's generate method (assuming it exists and handles generation)
                    # Pass necessary generation parameters (e.g., start token, max length)
                    # We might need to get start_token_id similar to how it was done before
                    start_token_id = self.tokenizer.bos_token_id if self.tokenizer.bos_token_id is not None else self.tokenizer.cls_token_id
                    if start_token_id is None: # Fallback if no BOS/CLS
                        start_token_id = self.tokenizer.pad_token_id
                        logger.warning(f"No BOS or CLS token found. Using PAD token ({start_token_id}) as start token for generation.")
                        
                    # Assuming decoder has a method like `generate` or similar
                    # The TextDecoder class might need its own generate method wrapping the underlying model's generate
                    generated_ids = self.decoder.generate(
                        latent_vector=fake_latents, 
                        max_new_tokens=max_length - 1, # Match TextDecoder param, adjust for start token
                        start_token_id=start_token_id # Match TextDecoder param
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

# Helper function (outside class)
def calculate_distinct_n(sentences: list, n: int):
    """Calculates distinct-n score for a list of sentences."""
    if not sentences:
        return 0.0
    
    all_ngrams = []
    total_ngrams_count = 0
    
    for sentence in sentences:
        tokens = nltk.word_tokenize(sentence.lower()) # Tokenize and lowercase
        if not tokens: continue # Skip empty sentences after tokenization
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
        return 0.0 # Cannot compute Self-BLEU with less than 2 sentences

    total_bleu = 0.0
    smoothing_function = SmoothingFunction().method1 # Choose a smoothing method
    weights = tuple(1. / n for _ in range(n)) # Uniform weights for BLEU-n

    tokenized_sentences = [nltk.word_tokenize(s.lower()) for s in sentences]

    for i in range(len(tokenized_sentences)):
        hypothesis = tokenized_sentences[i]
        references = tokenized_sentences[:i] + tokenized_sentences[i+1:]
        if not hypothesis or not references: # Skip if hypothesis or all references are empty
            continue
        # sentence_bleu expects references as a list of lists of tokens
        # Here, references is already a list of lists of tokens
        try:
            score = sentence_bleu(references, hypothesis, weights=weights, smoothing_function=smoothing_function)
            total_bleu += score
        except ZeroDivisionError: # Handle cases where smoothing might not be enough
             pass # Assign BLEU score of 0 if division by zero occurs
        except Exception as e:
            logger.warning(f"Error during sentence_bleu calculation: {e}. Hyp: {hypothesis}, Ref sample: {references[0] if references else 'None'}")
            pass # Skip sentence pair on other errors

    # Average BLEU score over all sentences
    return total_bleu / len(tokenized_sentences) if tokenized_sentences else 0.0

# Example of how to run the trainer (usually called from main.py)
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Train Latent Space GAN')
    parser.add_argument('--config', type=str, required=True, help='Path to the configuration YAML file')
    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    trainer = Trainer(config_path=args.config)
    trainer.train()
