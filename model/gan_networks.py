import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm
import logging

logger = logging.getLogger(__name__)

class LatentGenerator(nn.Module):
    """MLP Generator for the Latent Space GAN.

    Takes a noise vector as input and outputs a fake latent embedding.
    """
    def __init__(self, noise_dim: int, latent_dim: int, hidden_dims: list[int]):
        super().__init__()
        self.noise_dim = noise_dim
        self.latent_dim = latent_dim
        
        layers = []
        input_d = noise_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(input_d, h_dim))
            layers.append(nn.LeakyReLU(0.2, inplace=True)) # Common activation for GANs
            # Consider adding BatchNorm1d here for stability
            # layers.append(nn.BatchNorm1d(h_dim))
            input_d = h_dim
            
        # Output layer
        layers.append(nn.Linear(input_d, latent_dim))
        # Usually no activation or BatchNorm on the GAN generator's output layer,
        # unless specific properties are desired (e.g., tanh for [-1, 1] range).
        # Assuming the latent space doesn't have a strict range.
        
        self.model = nn.Sequential(*layers)
        logger.info(f"Initialized LatentGenerator MLP: Noise({noise_dim}) -> Hidden{hidden_dims} -> Latent({latent_dim})")

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: Input noise tensor (shape: [batch_size, noise_dim]).

        Returns:
            Fake latent embedding tensor (shape: [batch_size, latent_dim]).
        """
        return self.model(z)


class LatentDiscriminator(nn.Module):
    """MLP Discriminator (Critic) for the Latent Space GAN.

    Takes a latent embedding (real or fake) as input and outputs a scalar score.
    Designed for WGAN-GP, so the output is unbounded (no final sigmoid).
    """
    def __init__(self, latent_dim: int, hidden_dims: list[int]):
        super().__init__()
        self.latent_dim = latent_dim
        
        layers = []
        input_d = latent_dim
        for h_dim in hidden_dims:
            layers.append(spectral_norm(nn.Linear(input_d, h_dim)))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            # LayerNorm or Dropout could be added here
            # layers.append(nn.LayerNorm(h_dim))
            # layers.append(nn.Dropout(0.3))
            input_d = h_dim
            
        # Output layer (scalar score)
        layers.append(spectral_norm(nn.Linear(input_d, 1)))
        # No activation function for WGAN discriminator output
        
        self.model = nn.Sequential(*layers)
        logger.info(f"Initialized LatentDiscriminator MLP: Latent({latent_dim}) -> Hidden{hidden_dims} -> Score(1)")

    def forward(self, latent_vector: torch.Tensor) -> torch.Tensor:
        """
        Args:
            latent_vector: Input latent embedding (real or fake).
                           Shape: [batch_size, latent_dim].

        Returns:
            Discriminator score (scalar) for each input vector.
            Shape: [batch_size, 1].
        """
        return self.model(latent_vector)

# Example usage (for testing)
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    # Config placeholders
    _noise_dim = 100
    _latent_dim = 384
    _hidden_dims_g = [512, 512]
    _hidden_dims_d = [512, 256]
    _batch_size = 4

    # Test Generator
    generator = LatentGenerator(noise_dim=_noise_dim, latent_dim=_latent_dim, hidden_dims=_hidden_dims_g)
    print("\nGenerator Architecture:")
    print(generator)
    test_noise = torch.randn(_batch_size, _noise_dim)
    fake_latent = generator(test_noise)
    logger.info(f"Generator Input Noise Shape: {test_noise.shape}")
    logger.info(f"Generator Output Fake Latent Shape: {fake_latent.shape}")
    assert fake_latent.shape == (_batch_size, _latent_dim)

    # Test Discriminator
    discriminator = LatentDiscriminator(latent_dim=_latent_dim, hidden_dims=_hidden_dims_d)
    print("\nDiscriminator Architecture:")
    print(discriminator)
    # Test with fake latent from generator
    score = discriminator(fake_latent)
    logger.info(f"Discriminator Input Latent Shape: {fake_latent.shape}")
    logger.info(f"Discriminator Output Score Shape: {score.shape}")
    assert score.shape == (_batch_size, 1)

    # Test with random real-like latent
    real_latent = torch.randn(_batch_size, _latent_dim)
    score_real = discriminator(real_latent)
    logger.info(f"Discriminator Input Real Latent Shape: {real_latent.shape}")
    logger.info(f"Discriminator Output Real Score Shape: {score_real.shape}")
    assert score_real.shape == (_batch_size, 1)

    logger.info("\nGAN Network tests passed!")
