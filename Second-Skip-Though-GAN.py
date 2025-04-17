import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
import random
import re  # Added import for regex
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from transformers.modeling_outputs import BaseModelOutput  # Add this import

# --- Hyperparameters ---
latent_dim = 768        # Dimension of sentence latent embeddings (e.g., BART encoder output size)
noise_dim = 100         # Dimension of random noise input to generator
code_disc_dim = 10      # Size of discrete latent code (InfoGAN c_discrete has 10 categories)
code_cont_dim = 4       # Dimension of continuous latent code (InfoGAN c_continuous)
batch_size = 32
lr = 1e-4               # Learning rate for Adam optimizers
beta1, beta2 = 0.0, 0.9 # Adam betas for WGAN (β1=0 to avoid momentum on first moment)
lambda_gp = 10          # Gradient penalty coefficient for WGAN-GP
lambda_info = 1.0       # Weight for InfoGAN mutual information loss in generator
n_critic = 5            # Number of discriminator updates per generator update (WGAN training)

# --- Device configuration ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# --- Transformer-based Sentence Autoencoder (Encoder/Decoder) ---
# Assume we have a transformer encoder and decoder for sentences.
# For example, we might use a BART encoder for embeddings and a custom transformer decoder.
# Here, we use a pre-trained BART encoder for embedding extraction and its decoder for generation (frozen during GAN training).
from transformers import BartTokenizer, BartForConditionalGeneration
tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
bart_model = BartForConditionalGeneration.from_pretrained("facebook/bart-base").to(device)
bart_model.eval()  # freeze BART parameters

def get_sentence_embedding(text: str) -> torch.Tensor:
    """Encode a sentence into a latent vector using BART's encoder (or custom transformer encoder)."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
    with torch.no_grad():
        encoder_outputs = bart_model.get_encoder()(inputs.input_ids, attention_mask=inputs.attention_mask)
    # Average-pool the encoder output to get a fixed-length sentence embedding
    sentence_embedding = encoder_outputs.last_hidden_state.mean(dim=1)  # shape: (1, latent_dim)
    return sentence_embedding

# --- Define Minibatch Discrimination Layer ---
class MinibatchDiscrimination(nn.Module):
    def __init__(self, in_features: int, out_features: int, kernel_dims: int):
        """
        Minibatch discrimination layer to help the discriminator detect if generator outputs in a batch are too similar.
        It introduces features that measure differences among samples in the batch.
        Args:
            in_features: Number of input features for each sample (the dimension of the layer input).
            out_features: Number of features to output that capture batch-wide variation.
            kernel_dims: Dimension of the embedding space for computing sample distances.
        """
        super(MinibatchDiscrimination, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.kernel_dims = kernel_dims
        # Parameter tensor T of shape (in_features, out_features, kernel_dims)
        # Each out_feature will be computed using a kernel of dimension kernel_dims
        self.T = nn.Parameter(torch.randn(in_features, out_features, kernel_dims))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (N, A), where N = batch size, A = in_features
        N = x.size(0)
        # Compute matrices = x * T, shape (N, out_features * kernel_dims), then reshape
        matrices = x.mm(self.T.view(self.in_features, -1))
        matrices = matrices.view(N, self.out_features, self.kernel_dims)
        # Compute pairwise distances between samples in this transformed feature space
        M = matrices.unsqueeze(0)              # shape: (1, N, out_features, kernel_dims)
        M_T = matrices.unsqueeze(1)            # shape: (N, 1, out_features, kernel_dims)
        # L1 distance between each pair (i,j) for each out_feature:
        norm = torch.abs(M - M_T).sum(3)       # shape: (N, N, out_features)
        # Apply a negative exponential to distances to get similarity (larger if samples are closer)
        expnorm = torch.exp(-norm)            # shape: (N, N, out_features)
        # For each sample i, compute o_b[i] = sum_j exp(-||f_i - f_j||) over j (excluding itself)
        mask = torch.ones(N, N, device=x.device) - torch.eye(N, device=x.device)
        o_b = (expnorm * mask.unsqueeze(2)).sum(1)  # shape: (N, out_features)
        # Concatenate original features with minibatch features
        out = torch.cat([x, o_b], dim=1)      # shape: (N, in_features + out_features)
        return out

# --- Define Generator ---
class Generator(nn.Module):
    def __init__(self, noise_dim: int, code_disc_dim: int, code_cont_dim: int, latent_dim: int, hidden_dim: int = 512):
        """
        Generator network: takes random noise z and latent code c (both discrete and continuous parts) and outputs a latent embedding.
        Args:
            noise_dim: Dimensionality of input noise vector z.
            code_disc_dim: Size of discrete latent code (one-hot length).
            code_cont_dim: Dimension of continuous latent code.
            latent_dim: Dimension of the output latent embedding (should match encoder output size).
            hidden_dim: Hidden layer size for the generator MLP.
        """
        super(Generator, self).__init__()
        input_dim = noise_dim + code_disc_dim + code_cont_dim  # total input length
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(True),
            nn.Linear(hidden_dim * 2, latent_dim)
        )
    
    def forward(self, z: torch.Tensor, c_disc: torch.Tensor, c_cont: torch.Tensor) -> torch.Tensor:
        # Concatenate noise and code vectors
        zc = torch.cat([z, c_disc, c_cont], dim=1)  # shape: (batch_size, noise_dim + code_disc_dim + code_cont_dim)
        embedding = self.model(zc)  # output shape: (batch_size, latent_dim)
        return embedding

# --- Define Discriminator (Critic) ---
class Discriminator(nn.Module):
    def __init__(self, latent_dim: int, hidden_dim: int = 512, mb_out: int = 32, mb_kernel_dims: int = 16):
        """
        Discriminator (critic) network: takes a latent embedding and outputs a score (the Wasserstein critic value).
        Incorporates minibatch discrimination to examine batch diversity.
        Args:
            latent_dim: Dimension of input latent embeddings.
            hidden_dim: Hidden layer size for MLP.
            mb_out: Number of minibatch discrimination features to produce.
            mb_kernel_dims: Dimension of kernel for minibatch discrimination.
        """
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(latent_dim, hidden_dim * 2)
        self.fc2 = nn.Linear(hidden_dim * 2, hidden_dim)
        # Minibatch discrimination layer to generate mb_out features capturing batch statistics
        self.mb_discrimination = MinibatchDiscrimination(in_features=hidden_dim, out_features=mb_out, kernel_dims=mb_kernel_dims)
        # Final output layer: note input size is hidden_dim + mb_out (features from minibatch layer concatenated)
        self.fc_out = nn.Linear(hidden_dim + mb_out, 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Two-layer MLP with LeakyReLU activations
        h = F.leaky_relu(self.fc1(x), 0.2)
        h = F.leaky_relu(self.fc2(h), 0.2)
        # Apply minibatch discrimination to intermediate features h
        h_mb = self.mb_discrimination(h)
        # Output a single scalar (critic score) for each sample (no sigmoid, using WGAN-GP)
        out = self.fc_out(h_mb)
        return out

# --- Define Q-network for InfoGAN ---
class QNetwork(nn.Module):
    def __init__(self, latent_dim: int, hidden_dim: int = 256, code_disc_dim: int = 10, code_cont_dim: int = 4):
        """
        Auxiliary network Q for InfoGAN: tries to predict the latent code c from the generated embedding.
        Args:
            latent_dim: Dimension of input (latent embedding from generator).
            hidden_dim: Hidden size for Q network.
            code_disc_dim: Number of discrete code categories.
            code_cont_dim: Dimension of continuous code vector.
        """
        super(QNetwork, self).__init__()
        self.fc = nn.Linear(latent_dim, hidden_dim)
        self.fc_disc = nn.Linear(hidden_dim, code_disc_dim)   # predicts logits for discrete code categories
        self.fc_cont = nn.Linear(hidden_dim, code_cont_dim)   # predicts continuous code values
        # (We will apply softmax to disc logits for loss, and treat cont output as predicted mean)
    
    def forward(self, x: torch.Tensor):
        h = F.relu(self.fc(x))
        disc_logits = self.fc_disc(h)       # shape: (batch, code_disc_dim)
        cont_pred = torch.tanh(self.fc_cont(h))  # shape: (batch, code_cont_dim), tanh to bound in [-1,1]
        return disc_logits, cont_pred

# --- Initialize models ---
G = Generator(noise_dim, code_disc_dim, code_cont_dim, latent_dim).to(device)
D = Discriminator(latent_dim).to(device)
Q = QNetwork(latent_dim, code_disc_dim=code_disc_dim, code_cont_dim=code_cont_dim).to(device)

# Set optimizers. We combine G and Q parameters for a single optimizer since generator and Q are trained together.
optimizer_D = optim.Adam(D.parameters(), lr=lr, betas=(beta1, beta2))
optimizer_G = optim.Adam(list(G.parameters()) + list(Q.parameters()), lr=lr, betas=(beta1, beta2))

# --- Gradient Penalty function for WGAN-GP ---
def compute_gradient_penalty(D, real_samples: torch.Tensor, fake_samples: torch.Tensor) -> torch.Tensor:
    """Compute the gradient penalty for WGAN-GP. Ensures discriminator's gradient norm is close to 1."""
    batch_size = real_samples.size(0)
    # Random interpolate between real and fake samples
    alpha = torch.rand(batch_size, 1, device=device)
    alpha = alpha.expand_as(real_samples)
    interpolates = (alpha * real_samples + (1 - alpha) * fake_samples).requires_grad_(True)
    d_interpolates = D(interpolates)
    # For one-dimensional discriminator output, use ones as grad_outputs (since we take gradient of d_interpolates sum)
    fake = torch.ones(batch_size, 1, device=device)
    # Compute gradients of discriminator outputs w.r.t. interpolated inputs
    gradients = torch.autograd.grad(outputs=d_interpolates, inputs=interpolates,
                                    grad_outputs=fake, create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradients = gradients.view(batch_size, -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

# --- Dataset Preparation (load training sentences) ---
train_file_path = "Data/train.txt"
with open(train_file_path, "r", encoding="utf-8") as f:
    lines = f.readlines()
train_sentences = [line.strip() for line in lines if line.strip()]
print(f"Loaded {len(train_sentences)} training sentences.")

class SentenceDataset(Dataset):
    def __init__(self, sentences):
        self.sentences = sentences
    def __len__(self):
        return len(self.sentences)
    def __getitem__(self, idx):
        return self.sentences[idx]

dataset = SentenceDataset(train_sentences)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)

# --- Training Loop with WGAN-GP, InfoGAN, and Curriculum Learning ---
num_epochs = 20
for epoch in range(num_epochs):
    # Linearly anneal the mixing factor alpha from 0.5 (at epoch 0) down to 0 (by last epoch) for curriculum learning
    alpha = max(0.0, 0.5 * (1 - epoch / (num_epochs - 1)))
    for i, batch in enumerate(dataloader):
        # Prepare real sentence embeddings for this batch
        real_embeddings_list = []
        for text in batch:
            emb = get_sentence_embedding(text)  # (1, latent_dim)
            real_embeddings_list.append(emb)
        real_embeddings = torch.cat(real_embeddings_list, dim=0).to(device)  # shape: (batch_size, latent_dim)
        
        # Discriminator training (n_critic times per generator update)
        for _ in range(n_critic):
            # Sample random noise z and latent code c for each sample in batch
            z = torch.randn(real_embeddings.size(0), noise_dim, device=device)
            # Sample discrete latent code from uniform categorical distribution
            disc_labels = torch.randint(0, code_disc_dim, (real_embeddings.size(0),), device=device)
            c_disc = F.one_hot(disc_labels, num_classes=code_disc_dim).float()
            # Sample continuous latent code from uniform distribution in [-1,1]
            c_cont = (torch.rand(real_embeddings.size(0), code_cont_dim, device=device) * 2 - 1).float()
            # Generate fake latent embeddings from noise+code
            fake_embeddings = G(z, c_disc, c_cont)
            # Mix real embeddings with generated ones (curriculum learning)
            mixed_fake_embeddings = (1 - alpha) * fake_embeddings + alpha * real_embeddings
            # Compute discriminator outputs for real and fake
            real_scores = D(real_embeddings)
            fake_scores = D(mixed_fake_embeddings.detach())  # detach fake in D update to avoid backprop to G
            # Compute WGAN-GP discriminator loss: maximize D(real) - D(fake)
            # Equivalent to minimizing loss_D = -(D(real) - D(fake)) = -D(real) + D(fake)
            gp = compute_gradient_penalty(D, real_embeddings, mixed_fake_embeddings.detach())
            loss_D = -real_scores.mean() + fake_scores.mean() + lambda_gp * gp
            # Update Discriminator
            optimizer_D.zero_grad()
            loss_D.backward()
            optimizer_D.step()
        
        # Generator & Q-network training (after n_critic discriminator updates)
        # Sample a new set of noise and code for generator update
        z = torch.randn(real_embeddings.size(0), noise_dim, device=device)
        disc_labels = torch.randint(0, code_disc_dim, (real_embeddings.size(0),), device=device)
        c_disc = F.one_hot(disc_labels, num_classes=code_disc_dim).float()
        c_cont = (torch.rand(real_embeddings.size(0), code_cont_dim, device=device) * 2 - 1).float()
        fake_embeddings = G(z, c_disc, c_cont)
        mixed_fake_embeddings = (1 - alpha) * fake_embeddings + alpha * real_embeddings  # same mixing for G step
        # Discriminator output for fake (generator wants to maximize this, i.e. fool D)
        fake_scores = D(mixed_fake_embeddings)
        # Compute generator adversarial loss (Wasserstein generator loss = -D(fake))
        loss_G_adv = -fake_scores.mean()
        # Compute InfoGAN auxiliary losses:
        disc_logits, cont_pred = Q(mixed_fake_embeddings)
        # Discrete code loss: how well Q predicts the true discrete category (cross entropy)
        loss_Q_disc = F.cross_entropy(disc_logits, disc_labels)
        # Continuous code loss: mean squared error between predicted and true continuous code
        loss_Q_cont = F.mse_loss(cont_pred, c_cont)
        # Combine losses: Generator is trained to both fool D and maximize mutual information (minimize code prediction loss)
        loss_G = loss_G_adv + lambda_info * (loss_Q_disc + loss_Q_cont)
        # Update Generator and Q-network together
        optimizer_G.zero_grad()
        loss_G.backward()
        optimizer_G.step()
        
        # Logging
        if i % 50 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Batch {i}: loss_D = {loss_D.item():.4f}, loss_G = {loss_G.item():.4f} (Adv={loss_G_adv.item():.4f}, Q_disc={loss_Q_disc.item():.4f}, Q_cont={loss_Q_cont.item():.4f})")

# --- Text Generation from Latent Embedding ---
def generate_text_from_latent(latent_embedding: torch.Tensor, max_length: int = 50, 
                             num_beams: int = 5, temperature: float = 0.7, 
                             top_k: int = 50, top_p: float = 0.95) -> str:
    """
    Generate a text sequence from a latent embedding using the BART decoder with improved
    text quality through better sampling strategies and post-processing cleanup.
    
    Args:
        latent_embedding: Latent embedding tensor from generator, shape (batch_size, latent_dim)
        max_length: Maximum length of the generated sequence
        num_beams: Number of beams for beam search (higher = more focused but less creative)
        temperature: Temperature for sampling (higher = more random, lower = more deterministic)
        top_k: Keep only the k most likely tokens at each step
        top_p: Keep tokens with cumulative probability >= top_p (nucleus sampling)
    
    Returns:
        Generated text as a string
    """
    latent_embedding = latent_embedding.to(device)
    # Create a dummy input token for BART
    dummy_input = tokenizer("", return_tensors="pt").to(device)
    
    with torch.no_grad():
        # Expand the latent embedding to have a sequence length dimension
        hidden_states = latent_embedding.unsqueeze(1).expand(-1, 20, -1)
        
        # Create a proper BaseModelOutput object that BART expects
        encoder_outputs = BaseModelOutput(
            last_hidden_state=hidden_states,
            # No need to provide hidden_states or attentions here
        )
        
        # Use BART's decoder with improved generation parameters
        outputs = bart_model.generate(
            input_ids=dummy_input.input_ids,
            encoder_outputs=encoder_outputs,  # Pass the BaseModelOutput object
            max_length=max_length,
            num_beams=num_beams,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            do_sample=True,
            no_repeat_ngram_size=3,
            early_stopping=True
        )
    
    # Decode the generated token IDs to text
    raw_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Clean the generated text to remove problematic patterns
    # Remove sequences of repeated punctuation
    cleaned_text = re.sub(r'([.,!?;:/-])\1+', r'\1', raw_text)
    
    # Remove isolated punctuation with spaces on both sides
    cleaned_text = re.sub(r'\s([.,!?;:/])\s', ' ', cleaned_text)
    
    # Fix spacing around apostrophes
    cleaned_text = re.sub(r'\s\'', '\'', cleaned_text)
    cleaned_text = re.sub(r'(\w)\'(\s)', r'\1\'\2', cleaned_text)
    
    # Remove multiple consecutive apostrophes or single quotes
    cleaned_text = re.sub(r'\'\'\'?|\'\'', '"', cleaned_text)
    
    # Clean up any remaining multiple spaces
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
    
    # If text is empty or just punctuation after cleaning, return a placeholder
    if not re.search(r'[a-zA-Z0-9]', cleaned_text):
        cleaned_text = "Generated text contained only punctuation or special characters."
    
    return cleaned_text

# Example: Generate text from a random latent vector after training
z = torch.randn(1, noise_dim, device=device)
disc_label = torch.randint(0, code_disc_dim, (1,), device=device)
c_disc = F.one_hot(disc_label, num_classes=code_disc_dim).float()
c_cont = (torch.rand(1, code_cont_dim, device=device) * 2 - 1).float()
latent = G(z, c_disc, c_cont)
gen_text = generate_text_from_latent(latent)
print("Generated Text example:", gen_text)

# --- Evaluation Metrics: BLEU, Self-BLEU, Distinct-n ---
def calculate_sentence_bleu(reference: str, candidate: str) -> dict:
    """
    Calculate BLEU-1 to BLEU-4 for a single candidate sentence against a single reference sentence.
    Returns BLEU scores as a dictionary.
    """
    ref_tokens = nltk.word_tokenize(reference.lower())
    cand_tokens = nltk.word_tokenize(candidate.lower())
    smoothing = SmoothingFunction().method1  # smoothing to handle zero counts
    bleu1 = sentence_bleu([ref_tokens], cand_tokens, weights=(1, 0, 0, 0), smoothing_function=smoothing)
    bleu2 = sentence_bleu([ref_tokens], cand_tokens, weights=(0.5, 0.5, 0, 0), smoothing_function=smoothing)
    bleu3 = sentence_bleu([ref_tokens], cand_tokens, weights=(0.33, 0.33, 0.34, 0), smoothing_function=smoothing)
    bleu4 = sentence_bleu([ref_tokens], cand_tokens, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothing)
    return {'bleu-1': bleu1, 'bleu-2': bleu2, 'bleu-3': bleu3, 'bleu-4': bleu4}

def compute_diversity_metrics(sentences: list) -> dict:
    """
    Compute diversity metrics (Self-BLEU and Distinct-1,2) for a list of generated sentences.
    - Self-BLEU: measures similarity of each sentence to others (lower is more diverse).
    - Distinct-1, Distinct-2: fraction of unique unigrams and bigrams among all sentences.
    """
    # Tokenize all sentences
    tokenized_sentences = [nltk.word_tokenize(s.lower()) for s in sentences]
    # Compute Self-BLEU (average BLEU-4 of each sentence against all others as references)
    # We use BLEU-4 as a single measure of similarity.
    bleu_scores = []
    for i, tokens in enumerate(tokenized_sentences):
        if len(tokens) == 0:
            continue
        # Prepare references (all other sentences)
        references = [tok for j, tok in enumerate(tokenized_sentences) if j != i and len(tok) > 0]
        if not references:
            continue
        # Compute BLEU score of sentence i against all other generated sentences
        score = sentence_bleu(references, tokens, smoothing_function=SmoothingFunction().method1)
        bleu_scores.append(score)
    self_bleu = float(np.mean(bleu_scores)) if bleu_scores else 0.0
    # Compute Distinct-n
    all_tokens = [token for tokens in tokenized_sentences for token in tokens]
    total_unigrams = len(all_tokens)
    unique_unigrams = len(set(all_tokens))
    # Collect all bigrams from each sentence
    all_bigrams = []
    for tokens in tokenized_sentences:
        all_bigrams += list(zip(tokens, tokens[1:]))  # list of bigram tuples
    total_bigrams = len(all_bigrams)
    unique_bigrams = len(set(all_bigrams))
    distinct_1 = unique_unigrams / total_unigrams if total_unigrams > 0 else 0.0
    distinct_2 = unique_bigrams / total_bigrams if total_bigrams > 0 else 0.0
    return {'self_bleu': self_bleu, 'distinct-1': distinct_1, 'distinct-2': distinct_2}

def evaluate_model(generator: nn.Module, reference_sentences: list, num_samples: int = 100):
    """
    Generate `num_samples` sentences from the generator and evaluate average BLEU and diversity metrics.
    """
    generator.eval()
    generated_sentences = []
    reference_sample = random.sample(reference_sentences, min(num_samples, len(reference_sentences)))
    # Generate sentences
    for _ in range(num_samples):
        # Random latent code
        z = torch.randn(1, noise_dim, device=device)
        disc_label = torch.randint(0, code_disc_dim, (1,), device=device)
        c_disc = F.one_hot(disc_label, num_classes=code_disc_dim).float()
        c_cont = (torch.rand(1, code_cont_dim, device=device) * 2 - 1).float()
        latent = generator(z, c_disc, c_cont)
        text = generate_text_from_latent(latent)
        generated_sentences.append(text)
    # Compute BLEU scores against reference_sample (using each reference once for simplicity)
    bleu_scores = {'bleu-1': [], 'bleu-2': [], 'bleu-3': [], 'bleu-4': []}
    for ref, gen in zip(reference_sample, generated_sentences):
        scores = calculate_sentence_bleu(ref, gen)
        for k, v in scores.items():
            bleu_scores[k].append(v)
    avg_bleu = {k: float(np.mean(v)) if v else 0.0 for k, v in bleu_scores.items()}
    # Compute diversity metrics on generated outputs
    div_metrics = compute_diversity_metrics(generated_sentences)
    return {'average_bleu': avg_bleu, 'diversity': div_metrics, 'samples': generated_sentences}

# After training, evaluate the model
evaluation_results = evaluate_model(G, train_sentences, num_samples=100)
print("Average BLEU scores:", evaluation_results['average_bleu'])
print("Diversity metrics:", evaluation_results['diversity'])
