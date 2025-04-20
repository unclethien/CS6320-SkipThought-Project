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

# Placeholder for other decoding functions (e.g., beam search)
# def beam_search_decode(...):
#     pass

# Placeholder for a generation wrapper function
# def generate_sequence(model, input_embedding, tokenizer, max_len=50, decoding_config=None):
#     if decoding_config is None:
#         decoding_config = {}
#     # ... generation loop using sample_with_topk_topp or beam_search ...
#     pass
