"""
Data preprocessing script for Skip-Thought GAN
Cleans and structures training data to improve model performance
"""
import os
import re
import nltk
import argparse
import random
from nltk.tokenize import sent_tokenize, word_tokenize

# Ensure NLTK resources are available
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    print("Downloading NLTK punkt tokenizer...")
    nltk.download('punkt')

def clean_text(text):
    """Clean and normalize text"""
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    # Remove excessive punctuation (repeated more than once)
    text = re.sub(r'([.,!?;:]){2,}', r'\1', text)
    # Normalize quotes and dashes
    text = text.replace('"', '"').replace('"', '"').replace('—', '-')
    return text

def filter_sentences(sentences, min_length=3, max_length=50):
    """Filter sentences based on length criteria"""
    filtered = []
    for sent in sentences:
        words = word_tokenize(sent)
        if min_length <= len(words) <= max_length:
            filtered.append(sent)
    return filtered

def main():
    parser = argparse.ArgumentParser(description="Preprocess text data for Skip-Thought GAN")
    parser.add_argument("--input", type=str, default="Data/train.txt", help="Input text file")
    parser.add_argument("--output", type=str, default="Data/train_processed.txt", help="Output processed file")
    parser.add_argument("--min_length", type=int, default=3, help="Minimum sentence word length")
    parser.add_argument("--max_length", type=int, default=50, help="Maximum sentence word length")
    parser.add_argument("--split_ratio", type=float, default=0.1, 
                        help="Ratio of data to use for validation (0 to disable)")
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    print(f"Processing {args.input}...")
    
    # Read input file
    with open(args.input, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # Split into sentences
    raw_sentences = sent_tokenize(text)
    print(f"Found {len(raw_sentences)} raw sentences")
    
    # Clean sentences
    cleaned_sentences = [clean_text(s) for s in raw_sentences]
    
    # Filter by length
    filtered_sentences = filter_sentences(
        cleaned_sentences, 
        min_length=args.min_length,
        max_length=args.max_length
    )
    print(f"Keeping {len(filtered_sentences)} sentences after filtering")
    
    # Shuffle for better training
    random.shuffle(filtered_sentences)
    
    # Split data if requested
    if args.split_ratio > 0:
        split_idx = int(len(filtered_sentences) * (1 - args.split_ratio))
        train_sentences = filtered_sentences[:split_idx]
        val_sentences = filtered_sentences[split_idx:]
        
        # Save validation set
        val_output = args.output.replace(".txt", "_val.txt")
        with open(val_output, 'w', encoding='utf-8') as f:
            f.write('\n'.join(val_sentences))
        print(f"Saved {len(val_sentences)} validation sentences to {val_output}")
    else:
        train_sentences = filtered_sentences
    
    # Save processed training data
    with open(args.output, 'w', encoding='utf-8') as f:
        f.write('\n'.join(train_sentences))
    
    print(f"Saved {len(train_sentences)} processed training sentences to {args.output}")
    
    # Print sample
    print("\nSample processed sentences:")
    for s in train_sentences[:5]:
        print(f"- {s}")

if __name__ == "__main__":
    main()