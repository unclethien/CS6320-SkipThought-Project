import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    """Injects positional information into the input embeddings. BATCH_FIRST"""
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        # Shape: [max_len, 1, d_model] - Initialize assuming seq_len first
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        # Transpose pe to [1, max_len, d_model] for batch_first compatibility
        pe = pe.transpose(0, 1)
        self.register_buffer('pe', pe) # Shape: [1, max_len, d_model]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args: x: Tensor, shape [batch_size, seq_len, embedding_dim]"""
        # Add positional encoding to the input tensor x
        # self.pe is [1, max_len, d_model]. Select up to seq_len.
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class Generator(nn.Module):
    """Transformer-based Generator (Decoder) adapted for Language Modeling. BATCH_FIRST.

    Generates logits for the next token in a sequence.
    Uses multiple Transformer decoder layers.
    """
    def __init__(self, config: dict, tokenizer):
        """
        Args:
            config (dict): Configuration dictionary for the generator model.
                           Expected keys: embed_dim, hidden_dim, num_layers, nhead, dropout, max_seq_len.
            tokenizer: Tokenizer used for vocab size.
        """
        super(Generator, self).__init__()
        embed_dim = config.get('embed_dim', 768)
        hidden_dim = config.get('hidden_dim', 2048)
        num_layers = config.get('num_layers', 6)
        nhead = config.get('nhead', 8)
        dropout = config.get('dropout', 0.1)
        max_seq_len = config.get('max_seq_len', 512) # Match data loading max_length ideally
        vocab_size = tokenizer.vocab_size

        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.config = config # Store config if needed later

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        # PositionalEncoding now expects batch_first=True
        self.pos_encoder = PositionalEncoding(embed_dim, dropout, max_seq_len)

        # Define a single Transformer decoder layer with batch_first=True
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            batch_first=True # Input/Output shape: (batch, seq_len, features)
        )

        # Stack multiple decoder layers
        # TransformerDecoder itself also needs batch_first=True
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # Batch normalization layer applied to the output features of the Transformer
        # BN1d expects (N, C) or (N, C, L). Input is (N, L, C) with batch_first=True.
        # Apply BN over the embedding dim (C).
        self.bn = nn.BatchNorm1d(embed_dim)

        # Output projection layer to vocabulary size
        self.fc_out = nn.Linear(embed_dim, vocab_size)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None):
        """
        Forward pass for Language Modeling.

        Args:
            input_ids: Input token IDs.
                       Shape: [batch_size, seq_len].
            attention_mask: Mask indicating non-padding tokens (1) and padding tokens (0).
                            Shape: [batch_size, seq_len].

        Returns:
            Output logits over the vocabulary.
            Shape: [batch_size, seq_len, vocab_size].
        """
        # 1. Embed the input sequence & add positional encoding
        # input_ids shape: [batch_size, seq_len]
        tgt_emb = self.embedding(input_ids) * math.sqrt(self.embed_dim)
        # Add positional encoding: requires [batch, seq_len, embed_dim]
        tgt_emb = self.pos_encoder(tgt_emb)
        # Shape: [batch_size, seq_len, embed_dim]

        # 2. Prepare Masks for Transformer Decoder
        # Causal mask (tgt_mask) to prevent attending to future tokens
        tgt_len = input_ids.size(1)
        # _generate_square_subsequent_mask creates based on seq_len first dim
        # We need [tgt_len, tgt_len]
        causal_mask = self._generate_square_subsequent_mask(tgt_len).to(input_ids.device)

        # Padding mask (tgt_key_padding_mask) based on attention_mask
        # TransformerDecoder expects True where values should be *ignored*.
        # attention_mask has 1 for valid tokens, 0 for padding. Need inverse.
        if attention_mask is not None:
            # Shape: [batch_size, seq_len]
            padding_mask = (attention_mask == 0)
        else:
            padding_mask = None # Assume no padding if mask not provided

        # 3. Pass through Transformer Decoder
        # In decoder-only LM, tgt is used as both input and memory source.
        decoder_output = self.transformer_decoder(
            tgt=tgt_emb,
            memory=tgt_emb, # Use target embedding as memory
            tgt_mask=causal_mask,
            memory_mask=None, # No memory mask needed when memory=tgt
            tgt_key_padding_mask=padding_mask,
            memory_key_padding_mask=padding_mask # Use same padding mask for memory if memory=tgt
        )
        # Shape: [batch_size, seq_len, embed_dim]

        # 4. Apply Batch Normalization
        # Input is (N, L, C). BN1d needs (N, C) or (N, C, L).
        # Permute to (N, C, L), apply BN, permute back to (N, L, C).
        N, L, C = decoder_output.shape
        bn_input = decoder_output.permute(0, 2, 1) # Shape: [batch_size, embed_dim, seq_len]
        bn_output = self.bn(bn_input)
        decoder_output_bn = bn_output.permute(0, 2, 1) # Shape: [batch_size, seq_len, embed_dim]

        # 5. Project to vocabulary logits
        logits = self.fc_out(decoder_output_bn)
        # Shape: [batch_size, seq_len, vocab_size]

        return logits

    def _generate_square_subsequent_mask(self, sz: int) -> torch.Tensor:
        """Generates a square causal mask for the sequence.
           Mask shape: [sz, sz]. True values are masked.
        """
        mask = torch.triu(torch.ones(sz, sz), diagonal=1).bool()
        return mask # True elements will be ignored in attention
