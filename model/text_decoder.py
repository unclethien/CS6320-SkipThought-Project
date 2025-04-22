import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import logging

logger = logging.getLogger(__name__)

class PositionalEncoding(nn.Module):
    """Injects positional information into the input embeddings."""
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 500):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        # Register as buffer so it's part of the model state but not trained
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        # x is expected to be [seq_len, batch_size, d_model]
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class TextDecoder(nn.Module):
    """Transformer Decoder for generating text from a latent vector."""
    def __init__(self, 
                 latent_dim: int, 
                 vocab_size: int, 
                 d_model: int, 
                 n_head: int, 
                 n_layer: int, 
                 d_ff: int, 
                 max_length: int, 
                 dropout: float = 0.1,
                 pad_token_id: int = 0):
        super().__init__()

        if d_model != latent_dim:
            logger.warning(f"Decoder d_model ({d_model}) != latent_dim ({latent_dim}). Ensure projection if needed.")
            # If latent_dim != d_model, you might need a projection layer 
            # for the memory input to the TransformerDecoder.
            # self.memory_projection = nn.Linear(latent_dim, d_model)
        # else:
        #     self.memory_projection = nn.Identity()

        self.d_model = d_model
        self.max_length = max_length
        self.pad_token_id = pad_token_id
        self.vocab_size = vocab_size

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_length)
        
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, 
                                                 nhead=n_head, 
                                                 dim_feedforward=d_ff, 
                                                 dropout=dropout, 
                                                 batch_first=True) # Use batch_first=True
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_layer)
        
        self.output_projection = nn.Linear(d_model, vocab_size)

        self._init_weights()
        logger.info(f"Initialized TextDecoder: Latent({latent_dim}) -> Text(Vocab: {vocab_size}, MaxLen: {max_length})")

    def _init_weights(self):
        # Initialize weights for better convergence
        initrange = 0.1
        self.token_embedding.weight.data.uniform_(-initrange, initrange)
        self.output_projection.bias.data.zero_()
        self.output_projection.weight.data.uniform_(-initrange, initrange)

    def _generate_square_subsequent_mask(self, sz: int, device: torch.device) -> torch.Tensor:
        """Generate a square causal mask for the sequence. Masked positions are True.
           attend.
        """
        # Mask out subsequent positions (upper triangle) -> True means masked
        return torch.triu(torch.ones((sz, sz), dtype=torch.bool, device=device), diagonal=1)

    def forward(self, latent_vector: torch.Tensor, tgt_tokens: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for training (teacher forcing).

        Args:
            latent_vector: The conditioning latent vector.
                           Shape: [batch_size, latent_dim].
            tgt_tokens: Target token indices (shifted right, with start token).
                        Shape: [batch_size, seq_len].

        Returns:
            Output logits over the vocabulary.
            Shape: [batch_size, seq_len, vocab_size].
        """
        batch_size, seq_len = tgt_tokens.shape
        device = tgt_tokens.device

        # Create target mask and padding mask
        tgt_mask = self._generate_square_subsequent_mask(seq_len, device)
        # Pad mask should be True for padded tokens (pad_token_id)
        tgt_padding_mask = (tgt_tokens == self.pad_token_id)

        # Embed target tokens and add positional encoding
        # Note: TransformerDecoderLayer expects batch_first=True
        tgt_emb = self.token_embedding(tgt_tokens) * math.sqrt(self.d_model)
        # Positional encoding expects [seq_len, batch_size, d_model], but we use batch_first
        # Need to adjust PositionalEncoding or transpose input/output if sticking to original PE design
        # Let's adjust PE to handle batch_first input
        # For batch_first=True, input to PE should be [batch_size, seq_len, d_model]
        # Modify PE later if needed, for now assume it handles batch_first correctly or adjust here.
        
        # Workaround: Transpose for PE, then transpose back
        tgt_emb = tgt_emb.transpose(0, 1) # [seq_len, batch_size, d_model]
        tgt_emb = self.pos_encoder(tgt_emb)
        tgt_emb = tgt_emb.transpose(0, 1) # [batch_size, seq_len, d_model]
        
        # Prepare memory (latent vector needs to be expanded for sequence length)
        # TransformerDecoder expects memory shape [batch_size, memory_seq_len, d_model]
        # Here, memory_seq_len is 1 (we condition on a single vector)
        memory = latent_vector.unsqueeze(1) # Shape: [batch_size, 1, latent_dim]
        
        # If latent_dim != d_model, project memory
        # memory = self.memory_projection(memory)

        # Decode
        output = self.transformer_decoder(tgt=tgt_emb, 
                                         memory=memory,
                                         tgt_mask=tgt_mask,
                                         tgt_key_padding_mask=tgt_padding_mask,
                                         memory_key_padding_mask=None) # No padding in memory
        # Output shape: [batch_size, seq_len, d_model]

        # Project to vocabulary size
        logits = self.output_projection(output)
        # Output shape: [batch_size, seq_len, vocab_size]

        return logits

    @torch.no_grad()
    def generate(self, latent_vector: torch.Tensor, 
                 start_token_id: int, 
                 max_new_tokens: int = 50, 
                 temperature: float = 1.0, 
                 top_k: int = 0,
                 top_p: float = 1.0) -> torch.Tensor:
        """
        Generate text autoregressively from a latent vector.

        Args:
            latent_vector: Conditioning latent vector. Shape: [batch_size, latent_dim].
            start_token_id: The token ID to start generation with.
            max_new_tokens: Maximum number of tokens to generate.
            temperature: Softmax temperature for sampling.
            top_k: If > 0, performs top-k sampling.
            top_p: If < 1.0, performs nucleus (top-p) sampling.

        Returns:
            Generated token indices. Shape: [batch_size, generated_seq_len].
        """
        self.eval() # Set model to evaluation mode
        batch_size = latent_vector.size(0)
        device = latent_vector.device

        # Initialize generated sequence with the start token for each batch item
        generated_tokens = torch.full((batch_size, 1), start_token_id, dtype=torch.long, device=device)
        
        # Prepare memory (needs to be expanded for each step but remains constant)
        memory = latent_vector.unsqueeze(1) # Shape: [batch_size, 1, latent_dim]
        # memory = self.memory_projection(memory) # Project if needed
        
        for _ in range(max_new_tokens):
            # Get the current sequence length
            current_seq_len = generated_tokens.size(1)
            
            # Create masks for the current sequence length
            tgt_mask = self._generate_square_subsequent_mask(current_seq_len, device)
            # No padding in generated sequence initially
            tgt_padding_mask = torch.zeros_like(generated_tokens, dtype=torch.bool, device=device)

            # Embed the current sequence
            tgt_emb = self.token_embedding(generated_tokens) * math.sqrt(self.d_model)
            # --- Apply Positional Encoding (Batch First Handling) --- 
            tgt_emb = tgt_emb.transpose(0, 1) # [seq_len, batch_size, d_model]
            tgt_emb = self.pos_encoder(tgt_emb)
            tgt_emb = tgt_emb.transpose(0, 1) # [batch_size, seq_len, d_model]
            # -------------------------------------------------------
            
            # Get decoder output for the current sequence
            output = self.transformer_decoder(tgt=tgt_emb, 
                                             memory=memory,
                                             tgt_mask=tgt_mask,
                                             tgt_key_padding_mask=tgt_padding_mask)
            # output shape: [batch_size, current_seq_len, d_model]
            
            # Get logits for the *last* token in the sequence
            last_token_logits = self.output_projection(output[:, -1, :]) # Shape: [batch_size, vocab_size]
            
            # Apply temperature scaling
            if temperature != 1.0:
                last_token_logits = last_token_logits / temperature
            
            # Apply top-k filtering
            if top_k > 0:
                v, _ = torch.topk(last_token_logits, top_k)
                last_token_logits[last_token_logits < v[:, [-1]]] = -float('Inf')
            # Apply top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(last_token_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                # Remove tokens with cumulative probability above threshold
                sorted_mask = cumulative_probs > top_p
                sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
                sorted_mask[..., 0] = False
                # Scatter mask to original logits
                indices_to_remove = sorted_mask.scatter(1, sorted_indices, sorted_mask)
                last_token_logits[indices_to_remove] = -float('Inf')
            
            # Sample the next token
            probs = F.softmax(last_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1) # Shape: [batch_size, 1]
            
            # Append the sampled token to the generated sequence
            generated_tokens = torch.cat((generated_tokens, next_token), dim=1)
            
            # Optional: Stop generation if an EOS token is generated for all items in batch
            # if (next_token == eos_token_id).all():
            #     break

        return generated_tokens

# Example usage (for testing)
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    # --- Configuration (Matches default.yaml example) --- 
    _latent_dim = 384
    _vocab_size = 30522
    _d_model = 384
    _n_head = 6
    _n_layer = 4
    _d_ff = 1536
    _max_length = 64
    _dropout = 0.1
    _pad_token_id = 0 # Assuming padding token ID is 0
    _start_token_id = 101 # Example: [CLS] token for BERT-like tokenizer
    _batch_size = 4
    _seq_len_train = 30 # Example sequence length for training test
    _device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # -----------------------------------------------------

    logger.info(f"Testing TextDecoder on device: {_device}")

    decoder = TextDecoder(
        latent_dim=_latent_dim,
        vocab_size=_vocab_size,
        d_model=_d_model,
        n_head=_n_head,
        n_layer=_n_layer,
        d_ff=_d_ff,
        max_length=_max_length,
        dropout=_dropout,
        pad_token_id=_pad_token_id
    ).to(_device)

    print("\nDecoder Architecture:")
    print(decoder)

    # --- Test Training Forward Pass --- 
    logger.info("\nTesting Training Forward Pass...")
    test_latent_vector = torch.randn(_batch_size, _latent_dim, device=_device)
    # Dummy target tokens (batch_size, seq_len)
    test_tgt_tokens = torch.randint(low=1, high=_vocab_size, size=(_batch_size, _seq_len_train), device=_device)
    # Replace some with pad tokens
    test_tgt_tokens[:, -5:] = _pad_token_id 

    logits = decoder(test_latent_vector, test_tgt_tokens)
    logger.info(f"Input Latent Vector Shape: {test_latent_vector.shape}")
    logger.info(f"Input Target Tokens Shape: {test_tgt_tokens.shape}")
    logger.info(f"Output Logits Shape: {logits.shape}")
    assert logits.shape == (_batch_size, _seq_len_train, _vocab_size)
    logger.info("Training Forward Pass Test OK.")

    # --- Test Generation (Inference) --- 
    logger.info("\nTesting Generation (Inference)...")
    gen_latent_vector = torch.randn(_batch_size, _latent_dim, device=_device)
    _max_gen = 20
    generated_ids = decoder.generate(
        gen_latent_vector, 
        start_token_id=_start_token_id, 
        max_new_tokens=_max_gen, 
        temperature=0.8, 
        top_k=10,
        top_p=0.9
    )
    logger.info(f"Generation Input Latent Shape: {gen_latent_vector.shape}")
    logger.info(f"Generated Token IDs Shape: {generated_ids.shape}")
    # Shape should be [batch_size, 1 (start_token) + max_gen] or less if EOS is implemented
    assert generated_ids.shape[0] == _batch_size
    assert generated_ids.shape[1] <= (_max_gen + 1)
    assert generated_ids.device == torch.device(_device)
    logger.info(f"Generated Tokens (Batch Item 0): {generated_ids[0].tolist()}")
    logger.info("Generation Test OK.")

    logger.info("\nTextDecoder tests passed!")
