import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    """Injects positional information into the input embeddings."""
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args: x: Tensor, shape [seq_len, batch_size, embedding_dim]"""
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class Generator(nn.Module):
    """Transformer-based Generator (Decoder) with Batch Normalization.

    Generates a paraphrase sequence conditioned on an input encoding.
    Uses multiple Transformer decoder layers and applies Batch Normalization
    to stabilize training and improve quality.
    """
    def __init__(self, embed_dim: int, hidden_dim: int, vocab_size: int, num_layers: int = 3, nhead: int = 8, dropout: float = 0.1, max_seq_len: int = 100):
        """
        Args:
            embed_dim: Dimensionality of the token embeddings and model dimension.
            hidden_dim: Dimensionality of the feedforward network model in nn.TransformerDecoderLayer.
            vocab_size: Size of the output vocabulary.
            num_layers: Number of nn.TransformerDecoderLayer layers in the nn.TransformerDecoder.
            nhead: Number of heads in the multiheadattention models.
            dropout: Dropout value.
            max_seq_len: Maximum sequence length for positional encoding.
        """
        super(Generator, self).__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoder = PositionalEncoding(embed_dim, dropout, max_seq_len)

        # Define a single Transformer decoder layer
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            batch_first=False # Default: expects (seq_len, batch, embed_dim)
        )

        # Stack multiple decoder layers
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # Batch normalization layer applied to the output features of the Transformer
        # We need to handle the sequence length dimension. BN1d expects (N, C) or (N, C, L).
        # Here, output is (seq_len, batch, embed_dim). We apply BN over the embedding dim.
        # We'll reshape, apply BN, and reshape back.
        self.bn = nn.BatchNorm1d(embed_dim)

        # Output projection layer to vocabulary size
        self.fc_out = nn.Linear(embed_dim, vocab_size)

    def forward(self, memory: torch.Tensor, tgt: torch.Tensor, tgt_mask: torch.Tensor = None, tgt_key_padding_mask: torch.Tensor = None):
        """
        Forward pass of the Generator.

        Args:
            memory: The sequence from the encoder's output (e.g., VAEEncoder's z).
                    Expected shape: [memory_seq_len, batch_size, embed_dim].
                    Often memory_seq_len is 1 if using a single latent vector.
            tgt: The target sequence input to the decoder (e.g., shifted ground truth for training,
                 or previously generated tokens for inference).
                 Expected shape: [tgt_seq_len, batch_size]. (Token indices)
            tgt_mask: Mask to prevent attention to future positions.
                      Shape: [tgt_seq_len, tgt_seq_len].
            tgt_key_padding_mask: Mask to indicate padding tokens in the target sequence.
                                  Shape: [batch_size, tgt_seq_len].

        Returns:
            Output logits over the vocabulary.
            Shape: [tgt_seq_len, batch_size, vocab_size].
        """
        # 1. Embed the target sequence & add positional encoding
        # Target shape: [tgt_seq_len, batch_size]
        tgt_emb = self.embedding(tgt) * math.sqrt(self.embed_dim)
        # Add positional encoding: requires [seq_len, batch, embed_dim]
        tgt_emb = self.pos_encoder(tgt_emb)
        # Shape: [tgt_seq_len, batch_size, embed_dim]

        # 2. Pass through Transformer Decoder
        # Ensure memory is in correct shape: [memory_seq_len, batch, embed_dim]
        # If memory is just [batch, embed_dim], unsqueeze it: memory.unsqueeze(0)
        if memory.dim() == 2:
            memory = memory.unsqueeze(0) # Now shape [1, batch, embed_dim]

        # Generate causal mask if not provided (for autoregressive decoding)
        if tgt_mask is None:
            tgt_len = tgt.size(0)
            tgt_mask = self._generate_square_subsequent_mask(tgt_len).to(tgt.device)

        decoder_output = self.transformer_decoder(
            tgt=tgt_emb,
            memory=memory,
            tgt_mask=tgt_mask,
            memory_mask=None, # No mask for memory needed typically
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=None # Assuming memory is not padded
        )
        # Shape: [tgt_seq_len, batch_size, embed_dim]

        # 3. Apply Batch Normalization
        # BN1d expects input shape (N, C) or (N, C, L).
        # Our output is (L, N, C) where L=seq_len, N=batch, C=embed_dim.
        # We need to permute, apply BN, and permute back.
        L, N, C = decoder_output.shape
        bn_input = decoder_output.permute(1, 2, 0) # Shape: [batch_size, embed_dim, seq_len]
        bn_output = self.bn(bn_input)
        decoder_output_bn = bn_output.permute(2, 0, 1) # Shape: [seq_len, batch_size, embed_dim]

        # 4. Project to vocabulary logits
        logits = self.fc_out(decoder_output_bn)
        # Shape: [tgt_seq_len, batch_size, vocab_size]

        return logits

    def _generate_square_subsequent_mask(self, sz: int) -> torch.Tensor:
        """Generates a square mask for the sequence. The masked positions are filled with float('-inf').
           Unmasked positions are filled with float(0.0).
        """
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
