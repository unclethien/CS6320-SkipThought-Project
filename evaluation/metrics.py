from typing import List, Dict, Tuple
from collections import Counter
import nltk
from nltk.util import ngrams
import numpy as np
import torch
from utils.decoding import generate_sequence

# Consider using libraries like sacrebleu for more standardized BLEU
# from sacrebleu.metrics import BLEU
# Or Hugging Face datasets library
# import datasets

# Ensure NLTK data is available (run nltk.download('punkt') once if needed)
try:
    nltk.word_tokenize('test')
except LookupError:
    print("NLTK 'punkt' resource not found. Please run: nltk.download('punkt')")
    # You might want to raise an error or handle this differently

def distinct_n(sentences: List[List[str]], n: int) -> float:
    """Calculates Distinct-n score: ratio of unique n-grams to total n-grams.

    Args:
        sentences: A list of sentences, where each sentence is a list of tokens.
        n: The order of n-grams (e.g., 1 for unigrams, 2 for bigrams).

    Returns:
        The Distinct-n score (float).
    """
    if not sentences: 
        return 0.0
        
    total_ngrams_count = 0
    unique_ngrams = set()

    for tokens in sentences:
        if len(tokens) < n:
            continue
        # Generate n-grams for the current sentence
        current_ngrams = list(ngrams(tokens, n))
        unique_ngrams.update(current_ngrams)
        total_ngrams_count += len(current_ngrams)

    if total_ngrams_count == 0:
        return 0.0

    return len(unique_ngrams) / total_ngrams_count

def calculate_self_bleu(sentences: List[List[str]], n_gram: int = 4) -> float:
    """Calculates Self-BLEU score.

    Computes the average BLEU score of each sentence against all other sentences
    in the generated set.

    Args:
        sentences: A list of generated sentences (list of lists of tokens).
        n_gram: The maximum n-gram order for BLEU calculation (e.g., 4 for BLEU-4).

    Returns:
        The Self-BLEU score (float). Lower is more diverse.
    """
    if len(sentences) <= 1:
        return 0.0 # Cannot compute Self-BLEU with 0 or 1 sentence

    total_bleu_score = 0.0
    num_sentences = len(sentences)
    weights = tuple(1. / n_gram for _ in range(n_gram)) # Uniform weights for BLEU-n

    for i in range(num_sentences):
        hypothesis = sentences[i]
        references = [sentences[j] for j in range(num_sentences) if i != j]
        
        if not hypothesis or not references: continue

        # Use nltk's sentence_bleu
        try:
            # smoothing_function helps avoid zero scores for short sentences or few overlaps
            bleu_score = nltk.translate.bleu_score.sentence_bleu(
                references,
                hypothesis,
                weights=weights,
                smoothing_function=nltk.translate.bleu_score.SmoothingFunction().method1
            )
            total_bleu_score += bleu_score
        except ZeroDivisionError:
            # Handle cases where BLEU calculation fails (e.g., hypothesis is empty)
            pass 

    return total_bleu_score / num_sentences if num_sentences > 0 else 0.0


def calculate_standard_metrics(hypotheses: List[str], references_list: List[List[str]]) -> Dict[str, float]:
    """Calculates standard metrics like BLEU, ROUGE, METEOR.

    Args:
        hypotheses: List of generated sentences (strings).
        references_list: List where each element is a list of reference sentences (strings)
                         for the corresponding hypothesis.

    Returns:
        Dictionary containing scores for BLEU-4, ROUGE-L, METEOR.
    """
    results = {}
    
    # Tokenize for NLTK-based metrics
    hyp_tokens = [nltk.word_tokenize(h.lower()) for h in hypotheses]
    ref_tokens_list = [[nltk.word_tokenize(r.lower()) for r in refs] for refs in references_list]

    # --- BLEU-4 (using NLTK corpus_bleu) ---
    try:
        bleu4_score = nltk.translate.bleu_score.corpus_bleu(
            ref_tokens_list, 
            hyp_tokens, 
            weights=(0.25, 0.25, 0.25, 0.25), # BLEU-4 weights
            smoothing_function=nltk.translate.bleu_score.SmoothingFunction().method1
        )
        results['BLEU-4'] = bleu4_score * 100 # Often reported as 0-100 scale
    except Exception as e:
        print(f"Warning: Could not compute BLEU-4: {e}")
        results['BLEU-4'] = 0.0
        
    # --- ROUGE-L (Requires 'rouge-score' package: pip install rouge-score) ---
    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        rouge_l_scores = []
        for hyp, refs in zip(hypotheses, references_list):
            # ROUGE typically takes strings, calculate score per hypothesis against its refs
            # Take max ROUGE-L F1 score over multiple references if available
            max_f1 = 0.0
            if not refs: # Handle case with no references
                rouge_l_scores.append(0.0)
                continue
            for ref in refs:
                 score = scorer.score(ref, hyp)
                 max_f1 = max(max_f1, score['rougeL'].fmeasure)
            rouge_l_scores.append(max_f1)
        results['ROUGE-L'] = np.mean(rouge_l_scores) * 100
    except ImportError:
        print("Warning: 'rouge-score' package not found. Skipping ROUGE-L.")
        results['ROUGE-L'] = 0.0
    except Exception as e:
        print(f"Warning: Could not compute ROUGE-L: {e}")
        results['ROUGE-L'] = 0.0

    # --- METEOR (Requires NLTK data: nltk.download('wordnet')) ---
    try:
        meteor_scores = []
        for hyp, refs in zip(hyp_tokens, ref_tokens_list):
            if not refs: # Handle case with no references
                 meteor_scores.append(0.0)
                 continue
            # NLTK meteor_score expects single hypothesis list and list of reference lists
            # Calculate METEOR for each hyp against its refs, take max or average? NLTK computes against the best matching reference.
            score = nltk.translate.meteor_score.meteor_score(refs, hyp)
            meteor_scores.append(score)
        results['METEOR'] = np.mean(meteor_scores) * 100
    except LookupError:
        print("Warning: NLTK 'wordnet' resource not found for METEOR. Skipping.")
        results['METEOR'] = 0.0
    except Exception as e:
        print(f"Warning: Could not compute METEOR: {e}")
        results['METEOR'] = 0.0
        
    return results


def evaluate_model(generator, dataloader, eval_config, device, tokenizer):
    """Evaluates the model on a dataset using specified metrics.

    Args:
        generator: The trained Generator model.
        dataloader: DataLoader for the evaluation dataset.
        eval_config: Dictionary containing evaluation parameters (metrics, decoding settings).
        device: The device to run evaluation on ('cuda' or 'cpu').
        tokenizer: Tokenizer object to decode generated token IDs.

    Returns:
        Dictionary containing calculated metric scores.
    """
    generator.eval()

    all_hypotheses_str = []       # List of generated sentences (strings)
    all_references_str = []       # List of lists of reference sentences (strings)
    all_hypotheses_tokens = []    # List of generated sentences (tokens)

    decoding_params = eval_config.get('decoding', {})
    metrics_to_compute = eval_config.get('metrics', ['BLEU-4', 'ROUGE-L', 'METEOR', 'Distinct-1', 'Distinct-2', 'Self-BLEU'])

    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc="Evaluating")
        for batch in progress_bar:
            # Batch loading: source tokens and references
            source_tokens = batch['source_tokens'].to(device)  # [batch, seq_len]
            references = batch['references']  # List[List[str]]
            batch_size = source_tokens.size(0)

            # Compute memory by embedding source tokens and mean pooling
            # src_emb: [batch, seq_len, embed_dim]
            src_emb = generator.embedding(source_tokens)
            memory = src_emb.mean(dim=1)  # [batch, embed_dim]

            # Generate paraphrase token IDs per sample
            generated_ids_batch = []
            for i in range(batch_size):
                mem = memory[i:i+1]  # [1, embed_dim]
                gen_ids = generate_sequence(generator, mem, tokenizer, decoding_params, device)
                generated_ids_batch.append(gen_ids)

            # Decode generated IDs to strings and tokens
            for gen_ids in generated_ids_batch:
                hyp_str = tokenizer.decode(gen_ids, skip_special_tokens=True)
                hyp_tokens = tokenizer.convert_ids_to_tokens(gen_ids)
                # Basic cleaning might be needed for tokens
                hyp_tokens_clean = [t for t in hyp_tokens if t not in tokenizer.all_special_tokens]
                
                all_hypotheses_str.append(hyp_str)
                all_hypotheses_tokens.append(hyp_tokens_clean)
                
            all_references_str.extend(references) # Add list of references for this batch

    # --- Calculate Metrics --- 
    results = {}
    # Standard metrics (BLEU, ROUGE, METEOR)
    if any(m in metrics_to_compute for m in ['BLEU-4', 'ROUGE-L', 'METEOR']):
        standard_scores = calculate_standard_metrics(all_hypotheses_str, all_references_str)
        results.update(standard_scores)

    # Diversity metrics (Distinct-N)
    if 'Distinct-1' in metrics_to_compute:
        results['Distinct-1'] = distinct_n(all_hypotheses_tokens, 1)
    if 'Distinct-2' in metrics_to_compute:
        results['Distinct-2'] = distinct_n(all_hypotheses_tokens, 2)

    # Self-BLEU
    if 'Self-BLEU' in metrics_to_compute:
        results['Self-BLEU'] = calculate_self_bleu(all_hypotheses_tokens)
        
    # Filter results based on requested metrics
    final_results = {k: v for k, v in results.items() if k in metrics_to_compute}

    generator.train() # Set back to train mode
    return final_results
