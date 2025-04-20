import torch
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
import os
import yaml
from tqdm import tqdm

# Import models and losses
from model.encoder import VAEEncoder
from model.generator import Generator
from model.discriminator import InfoGAN_Discriminator
from training.losses import compute_gradient_penalty, infogan_loss
# from data.dataset import ParaphraseDataLoader # Assuming a DataLoader exists
# from evaluation.metrics import evaluate_model # Assuming an evaluation function exists

class Trainer:
    """Handles the training loop for the VAE-InfoGAN Paraphrase model."""
    def __init__(self, encoder: VAEEncoder, generator: Generator, discriminator: InfoGAN_Discriminator, config: dict):
        self.encoder = encoder
        self.generator = generator
        self.discriminator = discriminator
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Move models to device
        self.encoder.to(self.device)
        self.generator.to(self.device)
        self.discriminator.to(self.device)

        # Optimizers (AdamW recommended)
        lr_g = config['training'].get('learning_rate_g', 1e-4)
        lr_d = config['training'].get('learning_rate_d', 1e-4)
        lr_q = config['training'].get('learning_rate_q', 1e-4) # For Q-network params within Discriminator
        lr_e = config['training'].get('learning_rate_e', lr_g) # Encoder LR, often tied to Generator

        # Separate params for Q-network if needed, otherwise train D params together
        q_params = [p for name, p in discriminator.named_parameters() if 'q_' in name]
        d_params = [p for name, p in discriminator.named_parameters() if 'q_' not in name]

        self.optimizer_E = optim.AdamW(self.encoder.parameters(), lr=lr_e)
        self.optimizer_G = optim.AdamW(self.generator.parameters(), lr=lr_g)
        self.optimizer_D = optim.AdamW(d_params, lr=lr_d)
        self.optimizer_Q = optim.AdamW(q_params, lr=lr_q)

        # Mixed Precision Scaler
        self.use_amp = config['training'].get('use_amp', True)
        self.scaler = GradScaler(enabled=self.use_amp and self.device.type == 'cuda')

        # Training settings
        self.num_epochs = config['training'].get('num_epochs', 50)
        self.critic_iterations = config['training'].get('critic_iterations', 5)
        self.lambda_gp = config['training'].get('wgan_lambda_gp', 10)
        self.lambda_kl = config['training'].get('kl_loss_weight', 1.0)
        self.lambda_info = config['training'].get('info_loss_weight', 1.0)

        self.output_dir = config.get('output_dir', 'results/default_run')
        os.makedirs(self.output_dir, exist_ok=True)

    def _train_discriminator_step(self, real_input_embed, real_paraphrase_embed, fake_paraphrase_embed, disc_code, cont_code):
        """One optimization step for the Discriminator (Critic) and Q-network."""
        self.optimizer_D.zero_grad()
        self.optimizer_Q.zero_grad()

        with autocast(enabled=self.use_amp and self.device.type == 'cuda'):
            # Detach fake samples to avoid grads flowing back to Generator during D update
            fake_paraphrase_embed = fake_paraphrase_embed.detach()

            # Critic scores for real and fake samples
            # Assuming D takes (condition, sample)
            real_score, _, _ = self.discriminator(real_input_embed, real_paraphrase_embed)
            fake_score, q_disc_logits_fake, q_cont_mu_fake = self.discriminator(real_input_embed, fake_paraphrase_embed)

            # Compute Gradient Penalty
            gp = compute_gradient_penalty(self.discriminator, real_paraphrase_embed, fake_paraphrase_embed, real_input_embed)

            # WGAN-GP Critic Loss
            critic_loss = fake_score.mean() - real_score.mean() + self.lambda_gp * gp

            # InfoGAN Q-Network Loss (calculated on fake samples)
            q_loss = infogan_loss(q_disc_logits_fake, disc_code, q_cont_mu_fake, cont_code)

            # Combined Discriminator/Q loss (train Q along with D)
            d_total_loss = critic_loss + self.lambda_info * q_loss

        # Backward pass and optimization step using scaler
        self.scaler.scale(d_total_loss).backward()
        self.scaler.step(self.optimizer_D)
        self.scaler.step(self.optimizer_Q)
        # Scaler update happens once per optimizer step set

        return critic_loss.item(), gp.item(), q_loss.item()

    def _train_generator_step(self, real_input_embed, fake_paraphrase_embed, disc_code, cont_code, mu, logvar):
        """One optimization step for the Generator and Encoder."""
        self.optimizer_E.zero_grad()
        self.optimizer_G.zero_grad()

        with autocast(enabled=self.use_amp and self.device.type == 'cuda'):
            # Get Discriminator score and Q-network predictions for fake samples
            fake_score, q_disc_logits_fake, q_cont_mu_fake = self.discriminator(real_input_embed, fake_paraphrase_embed)

            # WGAN Generator Adversarial Loss (maximize score on fake data -> minimize negative score)
            g_adv_loss = -fake_score.mean()

            # InfoGAN Q-Network Loss for Generator (maximize mutual information -> minimize negative Q loss)
            # Note: We want Generator to produce samples where Q can recover the codes.
            # Minimizing Q loss wrt G params achieves this.
            g_info_loss = infogan_loss(q_disc_logits_fake, disc_code, q_cont_mu_fake, cont_code)

            # VAE KL Divergence Loss (from Encoder)
            kl_loss = self.encoder.compute_kl_loss(mu, logvar)

            # Combined Generator/Encoder Loss
            g_total_loss = g_adv_loss + self.lambda_info * g_info_loss + self.lambda_kl * kl_loss

        # Backward pass and optimization step using scaler
        self.scaler.scale(g_total_loss).backward()
        self.scaler.step(self.optimizer_E)
        self.scaler.step(self.optimizer_G)
        # Scaler update happens once per optimizer step set

        return g_adv_loss.item(), g_info_loss.item(), kl_loss.item()

    def train(self, train_loader, val_loader): # Pass dataloaders
        """Main training loop."""
        print(f"Starting training on {self.device}...")
        best_val_score = -float('inf')
        epochs_no_improve = 0
        patience = self.config['training'].get('early_stopping_patience', 5)

        for epoch in range(self.num_epochs):
            self.encoder.train()
            self.generator.train()
            self.discriminator.train()
            
            g_losses, d_losses, kl_losses, info_losses, gp_losses = [], [], [], [], []
            
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.num_epochs}")
            for i, batch in enumerate(progress_bar):
                # TODO: Adapt batch loading based on actual dataset structure
                # Example assumes batch provides source embeddings, target embeddings, 
                # target token sequences, and latent codes if using InfoGAN
                source_tokens = batch['source_tokens'].to(self.device)
                target_tokens = batch['target_tokens'].to(self.device) # For teacher forcing G
                target_padding_mask = batch['target_padding_mask'].to(self.device)
                # Assume precomputed embeddings or pass tokens to an embedder first
                real_input_embed_src = batch['source_embedding'].to(self.device)
                real_paraphrase_embed_tgt = batch['target_embedding'].to(self.device)
                # Sample latent codes for InfoGAN
                disc_code = batch['discrete_code'].to(self.device)
                cont_code = batch['continuous_code'].to(self.device)

                # --- Train Discriminator --- 
                # Requires real inputs, real paraphrases, and fake paraphrases
                with torch.no_grad(): # Don't track grads for E/G when preparing D inputs
                    with autocast(enabled=self.use_amp and self.device.type == 'cuda'):
                        # Encode source sentence
                        z_enc, mu, logvar = self.encoder(real_input_embed_src)
                        # TODO: Generator needs latent codes + target sequence input
                        # Combine z_enc, disc_code, cont_code as 'memory' for Generator?
                        # Generator's forward needs adjustment to take codes.
                        # For simplicity, let's assume G takes z_enc and target tokens
                        # And D takes z_enc and generated *embeddings* (or features)
                        
                        # Generate fake paraphrase sequence logits (using teacher forcing or autoregressive)
                        # Simplified: assume G outputs embeddings directly for D input
                        fake_paraphrase_embed = self.generator(z_enc, target_tokens, tgt_key_padding_mask=target_padding_mask) 
                        # Need to handle Generator output: logits -> embeddings? Average pool? Last hidden state?
                        # Placeholder: Assume generator output is suitable for D after some processing
                        # Example: Take mean of output embeddings along seq len
                        if fake_paraphrase_embed.dim() == 3: # If [seq, batch, dim]
                             fake_paraphrase_embed = fake_paraphrase_embed.mean(dim=0)

                # Run D update step(s)
                d_loss_iter, gp_iter, q_loss_iter = self._train_discriminator_step(
                    real_input_embed=z_enc.detach(), # Use encoded input as condition
                    real_paraphrase_embed=real_paraphrase_embed_tgt, # Use real target embed
                    fake_paraphrase_embed=fake_paraphrase_embed, # Use generated embed
                    disc_code=disc_code,
                    cont_code=cont_code
                )
                d_losses.append(d_loss_iter)
                gp_losses.append(gp_iter)
                info_losses.append(q_loss_iter) # Info loss is part of D update here

                # --- Train Generator (less frequently) --- 
                if (i + 1) % self.critic_iterations == 0:
                    # Generate new fake samples for G update (ensure fresh grads)
                    with autocast(enabled=self.use_amp and self.device.type == 'cuda'):
                         # Re-encode source sentence
                        z_enc, mu, logvar = self.encoder(real_input_embed_src)
                        # Generate fake paraphrase sequence/embedding
                        fake_paraphrase_embed = self.generator(z_enc, target_tokens, tgt_key_padding_mask=target_padding_mask)
                        if fake_paraphrase_embed.dim() == 3:
                             fake_paraphrase_embed = fake_paraphrase_embed.mean(dim=0)

                    g_adv_loss_iter, g_info_loss_iter, kl_loss_iter = self._train_generator_step(
                        real_input_embed=z_enc, # Use encoded input as condition
                        fake_paraphrase_embed=fake_paraphrase_embed,
                        disc_code=disc_code,
                        cont_code=cont_code,
                        mu=mu,
                        logvar=logvar
                    )
                    g_losses.append(g_adv_loss_iter)
                    kl_losses.append(kl_loss_iter)
                    # Info loss for G is minimized in _train_generator_step

                    # Update the scaler (once per complete D+G iteration)
                    self.scaler.update()
                    
                progress_bar.set_postfix({
                    'D_loss': f'{d_loss_iter:.3f}', 
                    'G_loss': f'{g_adv_loss_iter:.3f}' if g_losses else 'N/A',
                    'KL_loss': f'{kl_loss_iter:.3f}' if kl_losses else 'N/A',
                    'Q_loss': f'{q_loss_iter:.3f}',
                    'GP': f'{gp_iter:.3f}'
                })

            # --- End of Epoch --- 
            avg_d_loss = sum(d_losses) / len(d_losses) if d_losses else 0
            avg_g_loss = sum(g_losses) / len(g_losses) if g_losses else 0
            avg_kl_loss = sum(kl_losses) / len(kl_losses) if kl_losses else 0
            avg_info_loss = sum(info_losses) / len(info_losses) if info_losses else 0
            avg_gp = sum(gp_losses) / len(gp_losses) if gp_losses else 0
            print(f"Epoch {epoch+1} Summary: D_loss={avg_d_loss:.4f}, G_loss={avg_g_loss:.4f}, KL={avg_kl_loss:.4f}, Info={avg_info_loss:.4f}, GP={avg_gp:.4f}")

            # Validation step
            # val_score = evaluate_model(self.generator, self.encoder, val_loader, self.config['evaluation'], self.device)
            # print(f"Validation Score (e.g., BLEU+METEOR): {val_score:.4f}")
            val_score = 0 # Placeholder - implement evaluation

            # Early stopping and Checkpointing
            if val_score > best_val_score:
                print(f"Validation score improved ({best_val_score:.4f} -> {val_score:.4f}). Saving model...")
                best_val_score = val_score
                epochs_no_improve = 0
                # Save models
                torch.save(self.encoder.state_dict(), os.path.join(self.output_dir, 'best_encoder.pth'))
                torch.save(self.generator.state_dict(), os.path.join(self.output_dir, 'best_generator.pth'))
                torch.save(self.discriminator.state_dict(), os.path.join(self.output_dir, 'best_discriminator.pth'))
            else:
                epochs_no_improve += 1
                print(f"Validation score did not improve for {epochs_no_improve} epochs.")
                if epochs_no_improve >= patience:
                    print(f"Early stopping triggered after {epoch+1} epochs.")
                    break

        print("Training finished.")
