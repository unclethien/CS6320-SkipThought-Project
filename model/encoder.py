import torch
import torch.nn as nn
import torch.nn.functional as F

class VAEEncoder(nn.Module):
    """Variational Autoencoder (VAE) style encoder.

    Encodes input features into a latent distribution (mean and log variance)
    and samples a latent vector using the reparameterization trick.
    Designed to provide a regularized, smooth latent space for the input sentence.
    """
    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int):
        """
        Args:
            input_dim: Dimensionality of the input features (e.g., sentence embedding size).
            hidden_dim: Dimensionality of the hidden layer.
            latent_dim: Dimensionality of the latent space.
        """
        super(VAEEncoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        self.latent_dim = latent_dim

    def forward(self, input_features: torch.Tensor):
        """Forward pass to get latent vector and distribution parameters.

        Args:
            input_features: Input tensor (e.g., sentence embeddings, shape: [batch_size, input_dim]).

        Returns:
            z: Sampled latent vector (shape: [batch_size, latent_dim]).
            mu: Mean of the latent distribution (shape: [batch_size, latent_dim]).
            logvar: Log variance of the latent distribution (shape: [batch_size, latent_dim]).
        """
        # Encode input to hidden representation
        hidden = F.relu(self.fc1(input_features))

        # Get parameters of the latent Gaussian distribution
        mu = self.fc_mu(hidden)             # Predict mean
        logvar = self.fc_logvar(hidden)      # Predict log variance
        
        # Reparameterization trick: z = mu + sigma * epsilon
        std = torch.exp(0.5 * logvar)       # Calculate standard deviation
        epsilon = torch.randn_like(std)     # Sample from standard normal
        z = mu + std * epsilon              # Sample latent vector z
        
        return z, mu, logvar

    def compute_kl_loss(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Computes the KL divergence loss between the learned distribution N(mu, exp(logvar))
           and the prior N(0, I).

        Args:
            mu: Mean of the latent distribution.
            logvar: Log variance of the latent distribution.

        Returns:
            KL divergence loss (scalar tensor).
        """
        # Formula: KL(q || p) = -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
        # We average over the batch and sum over the latent dimensions
        kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        return kl_div.mean() # Average KL loss over the batch

# Note: In a real application, the input_dim would likely come from a pre-trained
# sentence encoder (like BERT, Sentence-BERT) or a trained RNN/Transformer encoder
# applied to tokenized input sentences. This simple MLP version assumes pre-computed
# sentence embeddings are provided.
