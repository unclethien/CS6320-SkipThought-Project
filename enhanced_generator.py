"""
Enhanced Generator module for Skip-Thought GAN that matches the architecture used in training
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class EnhancedGenerator(nn.Module):
    def __init__(self, noise_dim: int, code_disc_dim: int, code_cont_dim: int, latent_dim: int, hidden_dim: int = 512):
        """
        Enhanced Generator network with batch normalization and more layers to match the saved model architecture.
        This matches the structure in the saved checkpoint file for compatibility.
        
        Args:
            noise_dim: Dimension of the random noise input
            code_disc_dim: Dimension of the discrete code (one-hot)
            code_cont_dim: Dimension of the continuous code
            latent_dim: Output dimension (should match the BART embedding dimension)
            hidden_dim: Size of the hidden layers
        """
        super(EnhancedGenerator, self).__init__()
        
        input_dim = noise_dim + code_disc_dim + code_cont_dim
        
        # Create a sequential model with more layers and batch normalization
        self.model = nn.Sequential(
            # Layer 0-1: First linear layer + activation
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(True),
            
            # Layer 2-3: Second linear layer + activation
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(True),
            
            # Layer 4-5: First BatchNorm layer (these were in the saved checkpoint)
            nn.Linear(hidden_dim, 1024),
            nn.BatchNorm1d(1024),
            
            # Layer 6-7: Second BatchNorm layer
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            
            # Layer 8-9: Final output layer
            nn.Linear(1024, latent_dim),
            nn.Tanh()  # Tanh to constrain outputs to [-1, 1] range
        )
    
    def forward(self, z, c_disc, c_cont):
        """Forward pass through the generator"""
        # Concatenate the noise and code inputs
        x = torch.cat([z, c_disc, c_cont], dim=1)
        # Pass through the model
        return self.model(x)