import torch
import torch.nn.functional as F

def sample_with_topk_topp(logits, top_k=0, top_p=0.0, temperature=1.0):
    """Applies top-k and/or nucleus (top-p) filtering to logits and samples a token index.

    Args:
        logits: Raw, unnormalized scores for each token in the vocabulary (shape: [vocab_size]).
        top_k: If > 0, keep only top k tokens with highest probability.
        top_p: If > 0.0, keep the top tokens with cumulative probability >= top_p (nucleus sampling).
               Nucleus sampling is applied first if both top_k and top_p are set.
        temperature: Softmax temperature for scaling logits before sampling.

    Returns:
        The sampled token index (int).
    """
    if temperature <= 0:
        temperature = 1.0 # Avoid division by zero or negative temp

    # Apply temperature scaling
    logits = logits / temperature

    # Calculate probabilities
    probs = F.softmax(logits, dim=-1)

    # Apply top-p (nucleus) filtering
    if top_p > 0.0:
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

        # Find the smallest set of tokens whose cumulative probability exceeds top_p
        indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep the first token above the threshold
        indices_to_remove[..., 1:] = indices_to_remove[..., :-1].clone()
        indices_to_remove[..., 0] = 0 # Always keep the most probable token

        # Create a mask to remove tokens
        indices_masked = sorted_indices[indices_to_remove]
        # Create a mask tensor initialized to 1s
        mask = torch.ones_like(probs, dtype=torch.bool)
        # Set masked indices to 0
        mask[indices_masked] = 0

        # Apply the mask to probabilities
        probs = probs.masked_fill(~mask, 0.0)

        # Re-normalize the probabilities
        if torch.sum(probs) > 0:
            probs = probs / torch.sum(probs)
        else:
            # Handle case where all probabilities become zero (should be rare)
            # Fallback: sample uniformly from top-1 (greedy)
            probs = torch.zeros_like(probs)
            probs[torch.argmax(logits)] = 1.0


    # Apply top-k filtering (after top-p if used)
    if top_k > 0:
        # If top_p was applied, the number of valid tokens might be less than top_k
        effective_top_k = min(top_k, probs.shape[-1])
        if effective_top_k < probs.shape[-1]: # Check if filtering is needed
            # Get the top k probabilities and their indices
            topk_vals, topk_indices = torch.topk(probs, effective_top_k)
            # Create a mask to keep only the top k tokens
            topk_mask = torch.zeros_like(probs, dtype=torch.bool)
            topk_mask[topk_indices] = 1
            # Apply the mask
            probs = probs.masked_fill(~topk_mask, 0.0)
            # Re-normalize
            if torch.sum(probs) > 0:
                probs = probs / torch.sum(probs)
            else:
                # Fallback if all probs become zero
                probs = torch.zeros_like(probs)
                probs[torch.argmax(logits)] = 1.0

    # Sample from the final distribution
    # Use multinomial sampling which expects probabilities
    token_id = torch.multinomial(probs, num_samples=1)

    return token_id.item()

def generate_sequence(generator, memory, tokenizer, decoding_config: dict, device):
    """Generates token IDs using the generator and decoding settings."""
    method = decoding_config.get('method', 'greedy')
    max_length = decoding_config.get('max_length', 50)
    top_k = decoding_config.get('top_k', 0)
    top_p = decoding_config.get('top_p', 0.0)
    temperature = decoding_config.get('temperature', 1.0)
    # Initialize input IDs with CLS token
    cls_id = getattr(tokenizer, 'cls_token_id', None)
    sep_id = getattr(tokenizer, 'sep_token_id', None)
    if cls_id is None and hasattr(tokenizer, 'convert_tokens_to_ids'):
        cls_id = tokenizer.convert_tokens_to_ids('[CLS]')
    if sep_id is None and hasattr(tokenizer, 'convert_tokens_to_ids'):
        sep_id = tokenizer.convert_tokens_to_ids('[SEP]')
    # Start sequence: [cls]
    input_ids = torch.tensor([[cls_id]], dtype=torch.long, device=device)  # [seq_len=1, batch=1]
    generated = []
    for _ in range(max_length):
        # Create mask
        tgt_mask = generator._generate_square_subsequent_mask(input_ids.size(0)).to(device)
        # Get logits: [seq_len, batch, vocab_size]
        logits = generator(memory, input_ids, tgt_mask=tgt_mask)
        next_logits = logits[-1, 0]  # [vocab_size]
        if method == 'greedy':
            next_id = int(torch.argmax(next_logits).item())
        else:
            next_id = sample_with_topk_topp(next_logits, top_k=top_k, top_p=top_p, temperature=temperature)
        # Append and continue
        input_ids = torch.cat([input_ids, torch.tensor([[next_id]], device=device)], dim=0)
        generated.append(next_id)
        if sep_id is not None and next_id == sep_id:
            break
    return generated

# EOF: Added generate_sequence
