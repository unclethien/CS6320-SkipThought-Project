import torch
import torch.nn as nn
import torch.nn.functional as F

class MinibatchDiscrimination(nn.Module):
    """Minibatch Discrimination layer to encourage diversity in GANs.

    Computes features based on the L1 distance between samples within a batch.
    From Salimans et al. (2016) "Improved Techniques for Training GANs".
    """
    def __init__(self, in_features: int, out_features: int, kernel_dim: int):
        """
        Args:
            in_features: Number of input features from the previous layer.
            out_features: Number of discrimination kernels (output feature dimension).
            kernel_dim: Dimensionality of the intermediate space for distance calculation.
        """
        super(MinibatchDiscrimination, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.kernel_dim = kernel_dim

        # Weight tensor T: projects input features into the intermediate space
        # Shape: [in_features, out_features * kernel_dim]
        self.T = nn.Parameter(torch.Tensor(in_features, out_features * kernel_dim))
        nn.init.normal_(self.T, 0, 1) # Initialize weights

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor from the previous layer (shape: [batch_size, in_features]).

        Returns:
            Minibatch features tensor (shape: [batch_size, out_features]).
        """
        # x shape: [batch_size, in_features]
        batch_size = x.size(0)

        # Project input features using tensor T
        # M shape: [batch_size, out_features * kernel_dim]
        M = torch.matmul(x, self.T)

        # Reshape M into [batch_size, out_features, kernel_dim]
        M = M.view(batch_size, self.out_features, self.kernel_dim)

        # Calculate pairwise L1 distances between samples for each kernel
        # Unsqueeze M to enable broadcasting for pairwise subtraction
        # M.unsqueeze(1) -> [batch_size, 1, out_features, kernel_dim]
        # M.unsqueeze(0) -> [1, batch_size, out_features, kernel_dim]
        diff = M.unsqueeze(1) - M.unsqueeze(0)  # Shape: [batch_size, batch_size, out_features, kernel_dim]

        # Sum absolute differences over the kernel dimension
        l1_dist = torch.sum(torch.abs(diff), dim=3)  # Shape: [batch_size, batch_size, out_features]

        # Apply negative exponential to convert distances to similarities
        # Add small epsilon for numerical stability if needed (though exp handles 0)
        similarity = torch.exp(-l1_dist) # Shape: [batch_size, batch_size, out_features]

        # Sum similarities over all other samples j for each sample i
        # Subtract 1 to remove self-similarity (exp(0)=1)
        minibatch_features = torch.sum(similarity, dim=1) - 1 # Shape: [batch_size, out_features]

        return minibatch_features


class InfoGAN_Discriminator(nn.Module):
    """Discriminator incorporating Minibatch Discrimination and InfoGAN Q-network.

    Distinguishes real vs. fake paraphrases while also predicting latent codes
    used by the Generator (for InfoGAN mutual information maximization).
    Uses Minibatch Discrimination to encourage generator diversity.
    Assumes input is concatenation of condition (e.g., z_enc) and paraphrase features.
    """
    def __init__(self, input_dim: int, hidden_dim: int, disc_code_dim: int, cont_code_dim: int, mbd_out_features: int, mbd_kernel_dim: int):
        """
        Args:
            input_dim: Dimensionality of the combined input (e.g., encoded input + paraphrase features).
            hidden_dim: Dimensionality of the hidden layers.
            disc_code_dim: Number of categories for the discrete latent code (InfoGAN).
            cont_code_dim: Dimensionality of the continuous latent code (InfoGAN).
            mbd_out_features: Output features for the MinibatchDiscrimination layer.
            mbd_kernel_dim: Kernel dimension for the MinibatchDiscrimination layer.
        """
        super(InfoGAN_Discriminator, self).__init__()

        # Shared layers for discrimination and Q-network input
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        # Layer Normalization or BatchNorm could be added here for stability
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)

        # Minibatch Discrimination layer
        self.mb_disc = MinibatchDiscrimination(
            in_features=hidden_dim,
            out_features=mbd_out_features,
            kernel_dim=mbd_kernel_dim
        )

        # --- Discriminator Head (Real vs. Fake) ---
        # Takes hidden features + minibatch features as input
        self.fc_out_d = nn.Linear(hidden_dim + mbd_out_features, 1)

        # --- InfoGAN Q-Network Head (Latent Code Prediction) ---
        # Takes hidden features (before MBD) as input
        # Needs a small MLP potentially
        self.q_fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.q_ln = nn.LayerNorm(hidden_dim // 2)
        # Output layer for discrete code (logits for classification)
        self.q_disc_logits = nn.Linear(hidden_dim // 2, disc_code_dim)
        # Output layer for continuous code (predicted mean)
        # Assuming fixed variance for simplicity, otherwise predict logvar too
        self.q_cont_mu = nn.Linear(hidden_dim // 2, cont_code_dim)

    def forward(self, condition: torch.Tensor, paraphrase_features: torch.Tensor):
        """
        Forward pass for the combined Discriminator and Q-network.

        Args:
            condition: Conditioning tensor (e.g., output z from VAEEncoder).
                       Shape: [batch_size, condition_dim].
            paraphrase_features: Features of the real or fake paraphrase.
                                 Shape: [batch_size, paraphrase_feature_dim].

        Returns:
            d_score: Discriminator output score (real vs. fake). For WGAN, this is unbounded.
                     Shape: [batch_size, 1].
            q_disc_logits: Logits for the predicted discrete latent code.
                           Shape: [batch_size, disc_code_dim].
            q_cont_mu: Predicted mean for the continuous latent code.
                       Shape: [batch_size, cont_code_dim].
        """
        # Combine condition and features
        combined_input = torch.cat([condition, paraphrase_features], dim=1)
        
        # Shared layers
        h = F.relu(self.ln1(self.fc1(combined_input)))
        h = F.relu(self.ln2(self.fc2(h)))
        # h shape: [batch_size, hidden_dim]

        # --- Discriminator Path ---
        # Compute minibatch discrimination features
        mb_feats = self.mb_disc(h) # Shape: [batch_size, mbd_out_features]
        # Concatenate hidden features with minibatch features
        h_cat_d = torch.cat([h, mb_feats], dim=1) # Shape: [batch_size, hidden_dim + mbd_out_features]
        # Final discriminator score
        d_score = self.fc_out_d(h_cat_d)

        # --- Q-Network Path ---
        # Use the shared hidden features 'h' before MBD
        q_h = F.relu(self.q_ln(self.q_fc1(h)))
        # Predict discrete code logits
        q_disc_logits = self.q_disc_logits(q_h)
        # Predict continuous code mean
        q_cont_mu = self.q_cont_mu(q_h)
        # Note: For continuous codes, Q often predicts both mu and logvar,
        # and the loss uses a Gaussian NLL. Here we simplify to predict only mu.

        return d_score, q_disc_logits, q_cont_mu
