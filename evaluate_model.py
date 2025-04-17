"""
Advanced evaluation script for Skip-Thought GAN
Assesses text generation quality with multiple metrics
"""
import os
import torch
import json
import argparse
import numpy as np
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from collections import defaultdict
import random
import matplotlib.pyplot as plt
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# Ensure NLTK resources are available
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/wordnet')
except LookupError:
    print("Downloading required NLTK data...")
    nltk.download('punkt')
    nltk.download('wordnet')

def calculate_bleu_scores(references, candidates):
    """Calculate BLEU scores for each candidate against all references"""
    smoothing = SmoothingFunction().method1
    bleu_scores = {'bleu-1': [], 'bleu-2': [], 'bleu-3': [], 'bleu-4': []}
    
    for candidate in candidates:
        cand_tokens = nltk.word_tokenize(candidate.lower())
        if len(cand_tokens) < 2:  # Skip very short candidates
            continue
            
        # Tokenize references
        ref_tokens_list = [nltk.word_tokenize(ref.lower()) for ref in references]
        
        # Calculate BLEU-1,2,3,4
        bleu1 = sentence_bleu(ref_tokens_list, cand_tokens, weights=(1, 0, 0, 0), smoothing_function=smoothing)
        bleu2 = sentence_bleu(ref_tokens_list, cand_tokens, weights=(0.5, 0.5, 0, 0), smoothing_function=smoothing)
        bleu3 = sentence_bleu(ref_tokens_list, cand_tokens, weights=(0.33, 0.33, 0.34, 0), smoothing_function=smoothing)
        bleu4 = sentence_bleu(ref_tokens_list, cand_tokens, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothing)
        
        bleu_scores['bleu-1'].append(bleu1)
        bleu_scores['bleu-2'].append(bleu2)
        bleu_scores['bleu-3'].append(bleu3)
        bleu_scores['bleu-4'].append(bleu4)
        
    # Calculate average scores
    avg_bleu = {k: float(np.mean(v)) if v else 0.0 for k, v in bleu_scores.items()}
    return avg_bleu, bleu_scores

def calculate_meteor_scores(references, candidates):
    """Calculate METEOR scores for candidates against references"""
    scores = []
    for candidate in candidates:
        cand_tokens = nltk.word_tokenize(candidate.lower())
        
        # Calculate meteor against each reference and take the best score
        ref_scores = []
        for ref in references:
            ref_tokens = nltk.word_tokenize(ref.lower())
            try:
                score = meteor_score([ref_tokens], cand_tokens)
                ref_scores.append(score)
            except:
                continue
        
        if ref_scores:
            scores.append(max(ref_scores))
            
    avg_meteor = float(np.mean(scores)) if scores else 0.0
    return avg_meteor, scores

def calculate_diversity_metrics(sentences):
    """Calculate diversity metrics: distinct n-grams and self-BLEU"""
    if not sentences:
        return {
            'distinct-1': 0.0,
            'distinct-2': 0.0,
            'self_bleu': 0.0
        }
    
    # Tokenize sentences
    tokenized_sentences = [nltk.word_tokenize(s.lower()) for s in sentences]
    
    # Calculate distinct-1 and distinct-2
    all_unigrams = []
    all_bigrams = []
    
    for tokens in tokenized_sentences:
        all_unigrams.extend(tokens)
        if len(tokens) > 1:
            all_bigrams.extend(list(zip(tokens, tokens[1:])))
    
    distinct_1 = len(set(all_unigrams)) / max(len(all_unigrams), 1)
    distinct_2 = len(set(all_bigrams)) / max(len(all_bigrams), 1)
    
    # Calculate Self-BLEU (how similar generations are to each other)
    smoothing = SmoothingFunction().method1
    self_bleu_scores = []
    
    # If there are enough sentences to calculate self-BLEU
    if len(tokenized_sentences) > 1:
        for i, tokens in enumerate(tokenized_sentences):
            if len(tokens) < 2:
                continue
                
            # Use all other sentences as references
            references = [s for j, s in enumerate(tokenized_sentences) if j != i]
            
            # Calculate BLEU-4 against other sentences
            score = sentence_bleu(references, tokens, 
                                weights=(0.25, 0.25, 0.25, 0.25),
                                smoothing_function=smoothing)
            self_bleu_scores.append(score)
    
    avg_self_bleu = float(np.mean(self_bleu_scores)) if self_bleu_scores else 0.0
    
    return {
        'distinct-1': distinct_1,
        'distinct-2': distinct_2,
        'self_bleu': avg_self_bleu
    }

def calculate_perplexity(sentences, model, tokenizer, device):
    """Calculate average perplexity using GPT-2"""
    if not sentences:
        return float('inf'), []
        
    perplexities = []
    for sentence in sentences:
        # Tokenize and prepare input
        inputs = tokenizer(sentence, return_tensors="pt").to(device)
        
        # Get model prediction
        with torch.no_grad():
            outputs = model(**inputs, labels=inputs.input_ids)
            
        # Calculate perplexity
        loss = outputs.loss
        perplexity = torch.exp(loss).item()
        perplexities.append(perplexity)
        
    avg_perplexity = float(np.mean(perplexities))
    return avg_perplexity, perplexities

def plot_metrics(data, output_dir):
    """Create visualizations of evaluation metrics"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot BLEU scores
    plt.figure(figsize=(10, 6))
    bleu_metrics = ['bleu-1', 'bleu-2', 'bleu-3', 'bleu-4']
    plt.bar(bleu_metrics, [data['bleu'][k] for k in bleu_metrics])
    plt.title('BLEU Scores')
    plt.ylabel('Score')
    plt.savefig(os.path.join(output_dir, 'bleu_scores.png'))
    
    # Plot diversity metrics
    plt.figure(figsize=(10, 6))
    diversity_metrics = ['distinct-1', 'distinct-2', 'self_bleu']
    plt.bar(diversity_metrics, [data['diversity'][k] for k in diversity_metrics])
    plt.title('Diversity Metrics')
    plt.ylabel('Score')
    plt.savefig(os.path.join(output_dir, 'diversity_metrics.png'))
    
    # If we have parameter analysis data
    if 'parameter_analysis' in data:
        # Plot effect of temperature on BLEU-1
        temps = sorted(set(p['temperature'] for p in data['parameter_analysis']))
        temp_scores = defaultdict(list)
        
        for param in data['parameter_analysis']:
            temp_scores[param['temperature']].append(param['bleu-1'])
        
        plt.figure(figsize=(10, 6))
        plt.plot(temps, [np.mean(temp_scores[t]) for t in temps], marker='o')
        plt.title('Effect of Temperature on BLEU-1 Score')
        plt.xlabel('Temperature')
        plt.ylabel('Average BLEU-1')
        plt.savefig(os.path.join(output_dir, 'temperature_effect.png'))

def main():
    parser = argparse.ArgumentParser(description="Evaluate Skip-Thought GAN generated text")
    parser.add_argument("--samples_path", type=str, required=True,
                        help="Path to JSON file with generated samples")
    parser.add_argument("--reference_path", type=str, default="Data/train.txt",
                        help="Path to reference text file")
    parser.add_argument("--output_path", type=str, default="results/evaluation_results.json",
                        help="Path to save evaluation results")
    parser.add_argument("--plots_dir", type=str, default="results/plots",
                        help="Directory to save evaluation plots")
    parser.add_argument("--use_gpt2", action="store_true",
                        help="Use GPT-2 to calculate perplexity")
    parser.add_argument("--num_references", type=int, default=100,
                        help="Number of reference sentences to use")
    args = parser.parse_args()
    
    # Check that files exist
    if not os.path.exists(args.samples_path):
        raise FileNotFoundError(f"Samples file not found: {args.samples_path}")
    if not os.path.exists(args.reference_path):
        raise FileNotFoundError(f"Reference file not found: {args.reference_path}")
    
    # Load generated samples
    with open(args.samples_path, 'r', encoding='utf-8') as f:
        samples_data = json.load(f)
    
    # Extract generated texts
    if 'generated_samples' in samples_data:
        generated_texts = [sample['text'] for sample in samples_data['generated_samples']]
    else:
        generated_texts = []
        
    if not generated_texts:
        print("No generated samples found in the provided file.")
        return
    
    print(f"Evaluating {len(generated_texts)} generated samples")
    
    # Load reference texts
    with open(args.reference_path, 'r', encoding='utf-8') as f:
        reference_texts = [line.strip() for line in f if line.strip()]
    
    if len(reference_texts) > args.num_references:
        reference_texts = random.sample(reference_texts, args.num_references)
    
    print(f"Using {len(reference_texts)} reference texts for evaluation")
    
    # Initialize results dictionary
    results = {
        'num_samples': len(generated_texts),
        'num_references': len(reference_texts)
    }
    
    # Calculate BLEU scores
    print("Calculating BLEU scores...")
    avg_bleu, detailed_bleu = calculate_bleu_scores(reference_texts, generated_texts)
    results['bleu'] = avg_bleu
    
    # Calculate METEOR scores
    print("Calculating METEOR scores...")
    avg_meteor, meteor_scores = calculate_meteor_scores(reference_texts, generated_texts)
    results['meteor'] = avg_meteor
    
    # Calculate diversity metrics
    print("Calculating diversity metrics...")
    diversity_metrics = calculate_diversity_metrics(generated_texts)
    results['diversity'] = diversity_metrics
    
    # Calculate perplexity using GPT-2 if requested
    if args.use_gpt2:
        print("Calculating perplexity using GPT-2...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        gpt2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        gpt2_model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
        gpt2_model.eval()
        
        avg_perplexity, perplexities = calculate_perplexity(
            generated_texts, gpt2_model, gpt2_tokenizer, device
        )
        
        results['perplexity'] = {
            'average': avg_perplexity
        }
    
    # Parameter analysis if the data contains varying parameters
    if 'vary_params' in samples_data.get('parameters', {}) and samples_data['parameters']['vary_params']:
        print("Analyzing effect of generation parameters...")
        param_analysis = []
        
        # Extract parameter combinations and calculate metrics for each
        for sample in samples_data['generated_samples']:
            params = sample['params']
            param_result = {
                'temperature': params['temperature'],
                'top_p': params['top_p'],
                'num_beams': params['num_beams']
            }
            
            # Calculate individual BLEU scores
            for ref in random.sample(reference_texts, min(10, len(reference_texts))):
                ref_tokens = nltk.word_tokenize(ref.lower())
                sample_tokens = nltk.word_tokenize(sample['text'].lower())
                
                smoothing = SmoothingFunction().method1
                param_result['bleu-1'] = sentence_bleu(
                    [ref_tokens], sample_tokens, weights=(1, 0, 0, 0), smoothing_function=smoothing
                )
                
            param_analysis.append(param_result)
            
        results['parameter_analysis'] = param_analysis
    
    # Create output directory if needed
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    
    # Save results
    with open(args.output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
        
    print(f"Evaluation results saved to {args.output_path}")
    
    # Create plots
    print("Creating visualization plots...")
    plot_metrics(results, args.plots_dir)
    print(f"Plots saved to {args.plots_dir}")
    
    # Print summary
    print("\n--- Evaluation Summary ---")
    print(f"BLEU-1: {avg_bleu['bleu-1']:.4f}")
    print(f"BLEU-4: {avg_bleu['bleu-4']:.4f}")
    print(f"METEOR: {avg_meteor:.4f}")
    print(f"Distinct-1: {diversity_metrics['distinct-1']:.4f}")
    print(f"Distinct-2: {diversity_metrics['distinct-2']:.4f}")
    print(f"Self-BLEU: {diversity_metrics['self_bleu']:.4f}")
    if args.use_gpt2:
        print(f"Perplexity: {avg_perplexity:.2f}")

if __name__ == "__main__":
    main()