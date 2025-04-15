# Environment Setup and Imports
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
import random
from transformers import BartTokenizer, BartForConditionalGeneration
import os
import nltk
import ssl
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu, SmoothingFunction
import json
from datetime import datetime

# Setup SSL for nltk downloads (if needed)
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Download required nltk data
nltk.download('punkt_tab')

# Set random seeds for reproducibility
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

# CUDA Configuration - Force GPU usage if available
if not torch.cuda.is_available():
    print("WARNING: No CUDA device detected! This script is designed to run on GPU.")
    print("Training on CPU will be extremely slow. Please use a CUDA-enabled GPU.")
    response = input("Do you want to continue anyway? (y/n): ")
    if response.lower() != 'y':
        print("Exiting as requested.")
        exit()
    device = torch.device("cpu")
else:
    # Use GPU and show detailed information
    device = torch.device("cuda")
    torch.cuda.manual_seed_all(seed)
    # Force cuDNN to be used
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    
    # Print GPU information
    print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
    print(f"CUDA device count: {torch.cuda.device_count()}")
    print(f"CUDA version: {torch.version.cuda}")
    print(f"Available memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

print("Using device:", device)

# Load Pre-trained BART model and tokenizer for Transformer-based Sentence Embedding and Decoding
tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
bart_model = BartForConditionalGeneration.from_pretrained("facebook/bart-base").to(device)
bart_model.eval()  # Freeze weights during GAN training

# Function to extract sentence embeddings using BART's encoder
def get_sentence_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    # Force tensors to GPU
    for key in inputs:
        inputs[key] = inputs[key].to(device)
    with torch.no_grad():
        encoder_outputs = bart_model.model.encoder(inputs["input_ids"], attention_mask=inputs["attention_mask"])
    # Average pool across sequence length to get a single fixed-length vector
    sentence_embedding = encoder_outputs.last_hidden_state.mean(dim=1)  # shape: (batch_size, hidden_size)
    return sentence_embedding  # hidden_size=768 for bart-base

# Define Generator and Discriminator operating in the latent space (latent_dim = 768)
class Generator(nn.Module):
    def __init__(self, noise_dim=100, latent_dim=768, hidden_dim=512):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(noise_dim, hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(True),
            nn.Linear(hidden_dim * 2, latent_dim)
        )
    def forward(self, z):
        return self.model(z)

class Discriminator(nn.Module):
    def __init__(self, latent_dim=768, hidden_dim=512):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim, 1)  # Raw score for WGAN-GP; no sigmoid.
        )
    def forward(self, x):
        return self.model(x)

# Initialize Generator and Discriminator
noise_dim = 100
latent_dim = 768  # Must match BART's hidden size.
G = Generator(noise_dim=noise_dim, latent_dim=latent_dim).to(device)
D = Discriminator(latent_dim=latent_dim).to(device)

# Optimizers (using Adam with parameters tuned for WGAN-GP)
lr = 1e-4
optimizer_G = optim.Adam(G.parameters(), lr=lr, betas=(0.0, 0.9))
optimizer_D = optim.Adam(D.parameters(), lr=lr, betas=(0.0, 0.9))

# Define gradient penalty for WGAN-GP
def compute_gradient_penalty(D, real_samples, fake_samples):
    alpha = torch.rand(real_samples.size(0), 1).to(device)
    alpha = alpha.expand_as(real_samples)
    interpolates = (alpha * real_samples + (1 - alpha) * fake_samples).requires_grad_(True)
    d_interpolates = D(interpolates)
    fake_label = torch.ones(real_samples.size(0), 1).to(device)
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake_label,
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

# --- Dataset Preparation ---
# Load training sentences from "train.txt" (each line is a sentence)
train_file_path = "Data/train.txt"
with open(train_file_path, "r", encoding="utf-8") as f:
    train_sentences = f.readlines()
train_sentences = [line.strip() for line in train_sentences if line.strip()]
print(f"Loaded {len(train_sentences)} training sentences.")

# Define a custom Dataset class for sentences
class SentenceDataset(Dataset):
    def __init__(self, sentences):
        self.sentences = sentences
    def __len__(self):
        return len(self.sentences)
    def __getitem__(self, idx):
        return self.sentences[idx]

# Create the dataset and DataLoader
dataset = SentenceDataset(train_sentences)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=2)

# --- Training Loop for Transformer-Based Latent GAN using WGAN-GP ---
n_critic = 5  # Number of discriminator updates per generator update
lambda_gp = 10
num_epochs = 10

for epoch in range(num_epochs):
    for i, batch in enumerate(dataloader):
        # Convert each sentence in the batch to its BART-based embedding
        real_embeddings_list = []
        for text in batch:
            emb = get_sentence_embedding(text)  # Returns tensor of shape (1, latent_dim)
            real_embeddings_list.append(emb)
        real_embeddings = torch.cat(real_embeddings_list, dim=0)  # (batch_size, latent_dim)

        # Update Discriminator (critic) n_critic times
        for _ in range(n_critic):
            z = torch.randn(real_embeddings.size(0), noise_dim).to(device)
            fake_embeddings = G(z)
            real_validity = D(real_embeddings)
            fake_validity = D(fake_embeddings)
            gp = compute_gradient_penalty(D, real_embeddings, fake_embeddings)
            loss_D = -torch.mean(real_validity) + torch.mean(fake_validity) + lambda_gp * gp

            optimizer_D.zero_grad()
            loss_D.backward()
            optimizer_D.step()

        # Update Generator
        z = torch.randn(real_embeddings.size(0), noise_dim).to(device)
        fake_embeddings = G(z)
        fake_validity = D(fake_embeddings)
        loss_G = -torch.mean(fake_validity)

        optimizer_G.zero_grad()
        loss_G.backward()
        optimizer_G.step()

        if i % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Batch {i}: Loss_D = {loss_D.item():.4f}, Loss_G = {loss_G.item():.4f}")

# --- Text Generation ---
def generate_text_from_latent(latent_embedding, max_length=50):
    """
    Generate text from a latent embedding vector using BART's decoder.
    
    Args:
        latent_embedding: The latent vector from the Generator (shape: [1, latent_dim])
        max_length: Maximum length of the generated text
        
    Returns:
        generated_text: The generated text string
    """
    # We need to use the model differently - instead of directly using the generated latent embedding,
    # let's create a simple input and then inject our latent embedding into the model
    
    # Create dummy input
    dummy_input = tokenizer("", return_tensors="pt").to(device)
    
    # First encode with attention mask 
    with torch.no_grad():
        # Get the encoder outputs from a simple input
        encoder_outputs = bart_model.get_encoder()(
            dummy_input.input_ids, 
            attention_mask=dummy_input.attention_mask
        )
        
        # Replace the encoder's hidden states with our latent embedding (expanded to match expected dimensions)
        # Make sure dimensions match (batch_size, seq_length, hidden_size)
        seq_length = encoder_outputs[0].size(1)
        expanded_latent = latent_embedding.unsqueeze(1).expand(-1, seq_length, -1)
        encoder_outputs.last_hidden_state = expanded_latent
        
        # Now use the modified encoder outputs to generate text
        outputs = bart_model.generate(
            encoder_outputs=encoder_outputs,
            max_length=max_length,
            num_beams=5,
            early_stopping=True,
        )
        
    # Decode the output IDs to get the generated text
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text

# Generate and print an example text after training
z = torch.randn(1, noise_dim).to(device)
generated_latent = G(z)
output_text = generate_text_from_latent(generated_latent)
print("Generated Text:", output_text)

# --- BLEU Score Evaluation ---
def calculate_sentence_bleu(reference_sentence, candidate_sentence):
    """
    Calculate BLEU score for a single generated sentence against a reference sentence.
    
    Args:
        reference_sentence (str): The ground truth reference sentence
        candidate_sentence (str): The generated sentence to evaluate
        
    Returns:
        float: BLEU score (0-1)
    """
    # Tokenize the sentences
    reference_tokens = nltk.word_tokenize(reference_sentence.lower())
    candidate_tokens = nltk.word_tokenize(candidate_sentence.lower())
    
    # Calculate BLEU score with smoothing (to handle cases with no n-gram overlaps)
    smoothing = SmoothingFunction().method1
    
    # Calculate BLEU for different n-grams (1-4) with weights
    bleu1 = sentence_bleu([reference_tokens], candidate_tokens, weights=(1, 0, 0, 0), smoothing_function=smoothing)
    bleu2 = sentence_bleu([reference_tokens], candidate_tokens, weights=(0.5, 0.5, 0, 0), smoothing_function=smoothing)
    bleu3 = sentence_bleu([reference_tokens], candidate_tokens, weights=(0.33, 0.33, 0.34, 0), smoothing_function=smoothing)
    bleu4 = sentence_bleu([reference_tokens], candidate_tokens, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothing)
    
    return {
        'bleu-1': bleu1,
        'bleu-2': bleu2,
        'bleu-3': bleu3,
        'bleu-4': bleu4
    }

def evaluate_bleu_on_dataset(generator_model, reference_sentences, num_samples=100, batch_size=10):
    """
    Evaluate BLEU scores by generating multiple sentences and comparing to reference sentences.
    
    Args:
        generator_model (nn.Module): The trained Generator model
        reference_sentences (list): List of reference sentences to compare against
        num_samples (int): Number of samples to generate for evaluation
        batch_size (int): Number of samples to generate in each batch
        
    Returns:
        dict: Dictionary containing BLEU scores and generated samples
    """
    generator_model.eval()  # Set to evaluation mode
    
    all_bleu_scores = {
        'bleu-1': [],
        'bleu-2': [],
        'bleu-3': [],
        'bleu-4': []
    }
    
    generated_samples = []
    
    # Select a subset of reference sentences for evaluation if there are more than num_samples
    if len(reference_sentences) > num_samples:
        reference_subset = random.sample(reference_sentences, num_samples)
    else:
        reference_subset = reference_sentences
    
    print(f"\nEvaluating BLEU scores with {num_samples} samples...")
    
    for i in range(0, num_samples, batch_size):
        batch_end = min(i + batch_size, num_samples)
        batch_size_current = batch_end - i
        print(f"Generating samples {i+1}-{batch_end}...")
        
        # Generate random noise vectors
        z_batch = torch.randn(batch_size_current, noise_dim).to(device)
        
        # Generate latent embeddings
        with torch.no_grad():
            latent_embeddings = generator_model(z_batch)
        
        # Generate text from each latent embedding
        for j in range(batch_size_current):
            generated_text = generate_text_from_latent(latent_embeddings[j:j+1])
            reference_text = reference_subset[i+j]
            
            # Calculate BLEU scores
            bleu_scores = calculate_sentence_bleu(reference_text, generated_text)
            
            # Collect scores
            for key, score in bleu_scores.items():
                all_bleu_scores[key].append(score)
            
            # Save generated sample and reference for inspection
            generated_samples.append({
                'reference': reference_text,
                'generated': generated_text,
                'bleu-scores': bleu_scores
            })
    
    # Calculate average BLEU scores
    avg_bleu_scores = {key: sum(scores)/len(scores) for key, scores in all_bleu_scores.items()}
    
    # Create results dictionary
    results = {
        'average_scores': avg_bleu_scores,
        'samples': generated_samples,
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'num_samples': num_samples
    }
    
    return results

def save_bleu_results(results, filepath="bleu_evaluation_results.json"):
    """Save BLEU score evaluation results to a JSON file"""
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"BLEU evaluation results saved to {filepath}")

def print_bleu_summary(results):
    """Print a summary of the BLEU score evaluation results"""
    print("\n--- BLEU Score Evaluation Summary ---")
    print(f"Number of samples: {results['num_samples']}")
    print(f"Evaluation timestamp: {results['timestamp']}")
    print("\nAverage BLEU scores:")
    for key, score in results['average_scores'].items():
        print(f"{key}: {score:.4f}")
    
    # Print example comparisons
    print("\nExample comparisons (3 random samples):")
    samples = random.sample(results['samples'], min(3, len(results['samples'])))
    for i, sample in enumerate(samples):
        print(f"\nExample {i+1}:")
        print(f"Reference: {sample['reference']}")
        print(f"Generated: {sample['generated']}")
        print(f"BLEU-1: {sample['bleu-scores']['bleu-1']:.4f}, BLEU-2: {sample['bleu-scores']['bleu-2']:.4f}")
        print(f"BLEU-3: {sample['bleu-scores']['bleu-3']:.4f}, BLEU-4: {sample['bleu-scores']['bleu-4']:.4f}")

# Main section for running BLEU score evaluation
if __name__ == "__main__":
    # Check if an argument for testing mode is provided
    import sys
    
    # Create a directory for saving models and evaluation results if it doesn't exist
    os.makedirs('results', exist_ok=True)
    
    # If running in test mode, load the trained model and evaluate BLEU scores
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        print("Running in test mode - evaluating BLEU scores...")
        
        # Load trained model (assuming it's saved)
        model_path = 'results/skip_thought_gan_generator.pth'
        if os.path.exists(model_path):
            print(f"Loading trained model from {model_path}")
            G.load_state_dict(torch.load(model_path))
            G.eval()
        else:
            print(f"No saved model found at {model_path}. Using the current model state.")
        
        # Set number of samples to evaluate
        num_test_samples = 50
        
        # Run BLEU evaluation
        bleu_results = evaluate_bleu_on_dataset(G, train_sentences, num_samples=num_test_samples)
        
        # Save results
        save_bleu_results(bleu_results, filepath="results/bleu_evaluation_results.json")
        
        # Print summary
        print_bleu_summary(bleu_results)
    else:
        # Regular training run
        # Save the trained model
        model_path = 'results/skip_thought_gan_generator.pth'
        print(f"Saving trained Generator model to {model_path}")
        torch.save(G.state_dict(), model_path)
        
        print("\nTo evaluate BLEU scores on the trained model, run:")
        print("python Skip-Thought-GAN.py test")
