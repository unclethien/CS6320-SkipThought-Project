"""
Text generation script for Skip-Thought GAN
This script loads a trained model and generates text samples with different parameters
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import random
import json
import nltk
import sys
import re
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from transformers import BartTokenizer, BartForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutput
import numpy as np

# Add the current directory to sys.path to ensure module imports work
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import from our adapter module
from model_adapter import EnhancedGenerator, compute_diversity_metrics, calculate_sentence_bleu

# --- Default Hyperparameters ---
latent_dim = 768
noise_dim = 100
code_disc_dim = 10
code_cont_dim = 4
checkpoint_dir = "checkpoints"

# --- Parse command line arguments ---
parser = argparse.ArgumentParser(description="Generate text from Skip-Thought GAN")
parser.add_argument("--model_path", type=str, default=os.path.join(checkpoint_dir, "best_model.pth"),
                    help="Path to model checkpoint")
parser.add_argument("--num_samples", type=int, default=20, help="Number of samples to generate")
parser.add_argument("--max_length", type=int, default=50, help="Maximum length of generated text")
parser.add_argument("--num_beams", type=int, default=5, help="Number of beams for beam search")
parser.add_argument("--temperature", type=float, default=0.8, help="Temperature for sampling")
parser.add_argument("--top_k", type=int, default=50, help="Top-k sampling parameter")
parser.add_argument("--top_p", type=float, default=0.92, help="Nucleus sampling parameter")
parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
parser.add_argument("--save_path", type=str, default="results/generated_samples.json", 
                    help="Path to save generated samples")
parser.add_argument("--vary_params", action="store_true", 
                    help="Generate samples with varying temperature and top_p params")
parser.add_argument("--eval", action="store_true", 
                    help="Evaluate generated samples against training data")
args = parser.parse_args()

# --- Set random seed if provided ---
if args.seed is not None:
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

# --- Load model and setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load BART model for text generation
tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
bart_model = BartForConditionalGeneration.from_pretrained("facebook/bart-base").to(device)
bart_model.eval()

# Initialize Generator
G = EnhancedGenerator(noise_dim, code_disc_dim, code_cont_dim, latent_dim).to(device)

# Load checkpoint
print(f"Loading model from {args.model_path}")
if os.path.exists(args.model_path):
    checkpoint = torch.load(args.model_path, map_location=device)
    G.load_state_dict(checkpoint['G_state_dict'])
    print(f"Model loaded (trained for {checkpoint['epoch']} epochs)")
else:
    print(f"Warning: Checkpoint not found at {args.model_path}. Using untrained model.")

# --- Text generation function ---
def generate_text_from_latent(latent, max_length=50, num_beams=5, 
                             temperature=1.0, top_k=50, top_p=0.95):
    """
    Generate text from a latent embedding with improved cleaning functionality.
    """
    with torch.no_grad():
        # Prepare "dummy" input_ids
        dummy_input = tokenizer("", return_tensors="pt").to(device)
        
        # Create a proper BaseModelOutput object
        hidden_states = latent.unsqueeze(1).expand(-1, 20, -1)
        encoder_outputs = BaseModelOutput(
            last_hidden_state=hidden_states,
            # No need to provide hidden_states or attentions
        )
        
        # Use BART's decoder to generate
        outputs = bart_model.generate(
            input_ids=dummy_input.input_ids,
            encoder_outputs=encoder_outputs,  # Pass as BaseModelOutput object
            max_length=max_length,
            num_beams=num_beams,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            do_sample=True,
            no_repeat_ngram_size=3,
            early_stopping=True
        )
        
    # Decode to text
    raw_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Clean up the generated text
    # Remove sequences of repeated punctuation
    clean_text = re.sub(r'([.,!?;:/-])\1+', r'\1', raw_text)
    
    # Remove isolated punctuation with spaces on both sides
    clean_text = re.sub(r'\s([.,!?;:/])\s', ' ', clean_text)
    
    # Fix spacing around apostrophes
    clean_text = re.sub(r'\s\'', '\'', clean_text)
    clean_text = re.sub(r'(\w)\'(\s)', r'\1\'\2', clean_text)
    
    # Remove multiple consecutive apostrophes or single quotes
    clean_text = re.sub(r'\'\'\'?|\'\'', '"', clean_text)
    
    # Clean up any remaining multiple spaces
    clean_text = re.sub(r'\s+', ' ', clean_text).strip()
    
    # If text is empty or just punctuation after cleaning, return a placeholder
    if not re.search(r'[a-zA-Z0-9]', clean_text):
        clean_text = "Generated text contained only punctuation or special characters."
        
    return clean_text

# --- Load training data if needed for evaluation ---
train_sentences = []
if args.eval:
    train_file_path = "Data/train.txt"
    try:
        with open(train_file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        train_sentences = [line.strip() for line in lines if line.strip()]
        print(f"Loaded {len(train_sentences)} training sentences for evaluation.")
    except Exception as e:
        print(f"Error loading training data: {e}")
        print("Will proceed without evaluation.")

# --- Generate text samples ---
G.eval()
generated_samples = []

# Function to generate one sample with specific parameters
def generate_one_sample(temp, top_p_val, num_beams_val, disc_category=None):
    z = torch.randn(1, noise_dim, device=device)
    
    # Use specific discrete category if provided, otherwise random
    if disc_category is not None and 0 <= disc_category < code_disc_dim:
        disc_label = torch.tensor([disc_category], device=device)
    else:
        disc_label = torch.randint(0, code_disc_dim, (1,), device=device)
        
    c_disc = F.one_hot(disc_label, num_classes=code_disc_dim).float()
    c_cont = (torch.rand(1, code_cont_dim, device=device) * 2 - 1).float()
    
    latent = G(z, c_disc, c_cont)
    text = generate_text_from_latent(
        latent, 
        max_length=args.max_length, 
        num_beams=num_beams_val,
        temperature=temp, 
        top_k=args.top_k, 
        top_p=top_p_val
    )
    
    return {
        "text": text,
        "params": {
            "temperature": float(temp),
            "top_p": float(top_p_val),
            "num_beams": int(num_beams_val),
            "discrete_category": int(disc_label.item())
        }
    }

print("\n--- Generating Text Samples ---")

if args.vary_params:
    # Generate samples with varying parameters to find optimal settings
    temperatures = [0.6, 0.8, 1.0, 1.2]
    top_ps = [0.85, 0.9, 0.95, 0.98]
    num_beams_options = [1, 3, 5]
    
    for temp in temperatures:
        for top_p_val in top_ps:
            for num_beams_val in num_beams_options:
                sample = generate_one_sample(temp, top_p_val, num_beams_val)
                generated_samples.append(sample)
                print(f"\nTemperature: {temp}, Top-p: {top_p_val}, Num beams: {num_beams_val}")
                print(f"Generated: {sample['text']}")
else:
    # Generate samples with fixed parameters
    for i in range(args.num_samples):
        sample = generate_one_sample(args.temperature, args.top_p, args.num_beams)
        generated_samples.append(sample)
        print(f"\nSample {i+1}: {sample['text']}")

# --- Generate samples for each discrete category ---
print("\n--- Samples by Category ---")
category_samples = []
for category in range(code_disc_dim):
    sample = generate_one_sample(args.temperature, args.top_p, args.num_beams, disc_category=category)
    category_samples.append(sample)
    print(f"\nCategory {category}: {sample['text']}")

# --- Evaluate if requested ---
evaluation_results = {}
if args.eval and train_sentences:
    print("\n--- Evaluating Generated Samples ---")
    
    # Calculate BLEU scores against training data
    reference_sample = random.sample(train_sentences, min(len(generated_samples), len(train_sentences)))
    
    bleu_scores = {'bleu-1': [], 'bleu-2': [], 'bleu-3': [], 'bleu-4': []}
    for ref, gen in zip(reference_sample, [s["text"] for s in generated_samples]):
        scores = calculate_sentence_bleu(ref, gen)
        for k, v in scores.items():
            bleu_scores[k].append(v)
    
    avg_bleu = {k: float(np.mean(v)) for k, v in bleu_scores.items()}
    
    # Calculate diversity metrics
    div_metrics = compute_diversity_metrics([s["text"] for s in generated_samples])
    
    # Store evaluation results
    evaluation_results = {
        "average_bleu": avg_bleu,
        "diversity": div_metrics
    }
    
    print("Average BLEU scores:", avg_bleu)
    print("Diversity metrics:", div_metrics)

# --- Save results ---
if not os.path.exists(os.path.dirname(args.save_path)):
    os.makedirs(os.path.dirname(args.save_path))

results = {
    "generated_samples": generated_samples,
    "category_samples": category_samples,
    "parameters": vars(args)
}

if evaluation_results:
    results["evaluation"] = evaluation_results

with open(args.save_path, 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"\nResults saved to {args.save_path}")