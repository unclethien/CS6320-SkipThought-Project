Great! I’ll generate an improved version of your GAN-based paraphrasing system that focuses on both quality and diversity of outputs. This will include architecture-level upgrades, enhanced training strategies (curriculum, diversity-aware loss), and decoding improvements (like top-k/top-p sampling).

I'll also include Self-BLEU and Distinct-n metric implementations, and provide a detailed report summarizing the changes and expected impact.

I’ll get started and let you know once everything is ready.

```python
"""
# Paraphrase Generation GAN: Architecture and Training Enhancements

## Overview

We enhance a GAN-based paraphrase generation system (Transformer encoder-decoder in Skip-Thought style) 
to improve output *quality* and *diversity*. The following sections detail upgrades in **architecture**, 
**training**, **decoding**, and **evaluation**, as well as code refactoring and dataset improvements. 
Each improvement is explained along with its expected impact on the model's performance.

## Architecture Enhancements

### 1. Deeper Generator with Batch Normalization

We extend the Generator network with additional Transformer layers and apply Batch Normalization (BN) 
to stabilize training. Deeper layers enable the generator to model more complex mappings from input 
to paraphrase, improving fluency and fidelity. BN helps mitigate internal covariate shift and stabilize 
gradients ([[PDF] Generative Adversarial Networks: Part II - TJ Machine Learning Club](https://tjmachinelearning.com/lectures/1819/guest/gan2/GANS_II.pdf#:~:text=Another%20feature%20of%20DCGAN%20is,ous)), which is crucial in GAN training. The expected impact is a more **stable training** 
process and improved **generation quality** due to richer representation capacity.

Below is a snippet of an updated Generator with an extra Transformer decoder layer and BN applied:
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class Generator(nn.Module):
    def __init__(self, embed_dim, hidden_dim, vocab_size, num_layers=2):
        super(Generator, self).__init__()
        # Transformer-based decoder layers
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoding = nn.Parameter(torch.randn(1, 100, embed_dim))  # positional encodings (max length 100)
        # Stack multiple Transformer decoder layers for a deeper network
        decoder_layer = nn.TransformerDecoderLayer(d_model=embed_dim, nhead=8, dim_feedforward=hidden_dim)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        # Batch normalization layer to normalize decoder outputs (except final logits)
        self.bn = nn.BatchNorm1d(embed_dim)
        # Output projection to vocabulary size
        self.fc_out = nn.Linear(embed_dim, vocab_size)
    
    def forward(self, encoded_input, target_seq):
        """
        encoded_input: conditioning input embedding (from encoder or latent)
        target_seq: token indices of partial sequence to decode (for teacher forcing or autoregression)
        """
        # Embed target sequence tokens and add positional encoding
        tgt_emb = self.embedding(target_seq) + self.pos_encoding[:, :target_seq.size(1), :]
        # Transformer decoding (using encoded input as memory)
        # Transpose for batch-first to seq-first as needed by PyTorch Transformer
        memory = encoded_input.transpose(0, 1)  # memory shape: [seq_len_enc, batch, embed_dim]
        tgt = tgt_emb.transpose(0, 1)          # tgt shape: [seq_len_dec, batch, embed_dim]
        decoder_output = self.transformer_decoder(tgt, memory)  # shape: [seq_len_dec, batch, embed_dim]
        # Apply batch normalization to the output features (flatten seq and batch for BN, then reshape back)
        B, T = target_seq.size(0), target_seq.size(1)
        decoder_output_flat = decoder_output.transpose(0, 1).reshape(B * T, -1)
        decoder_output_bn = self.bn(decoder_output_flat).reshape(B, T, -1).transpose(0, 1)
        # Project to vocabulary logits
        logits = self.fc_out(decoder_output_bn)
        return logits

"""
In this design, `Generator` uses multiple Transformer decoder layers (`num_layers=2` by default, easily increased) 
and a BatchNorm layer (`self.bn`) applied to the decoder's output features. The BN normalizes across the mini-batch, 
which can **smooth optimization** and help avoid mode collapse in GAN training ([[PDF] Generative Adversarial Networks: Part II - TJ Machine Learning Club](https://tjmachinelearning.com/lectures/1819/guest/gan2/GANS_II.pdf#:~:text=Another%20feature%20of%20DCGAN%20is,ous)).

### 2. Minibatch Discrimination in the Discriminator

To combat GAN mode collapse (generator outputting low-diversity, repetitive paraphrases), we add *minibatch 
discrimination* to the Discriminator ([](http://arxiv.org/pdf/1606.03498#:~:text=what%20we%20call%20minibatch%20discrimination,than%20in%20isolation%2C%20could%20potentially)). This technique allows the discriminator to examine multiple 
samples at once and detect lack of diversity by measuring differences between samples in a batch ([](http://arxiv.org/pdf/1606.03498#:~:text=what%20we%20call%20minibatch%20discrimination,than%20in%20isolation%2C%20could%20potentially)). By 
penalizing the generator for producing identical or very similar outputs, we encourage more varied paraphrases 
(*mode diversity*). The expected impact is **higher output diversity** and reduced collapse to trivial rewrites.

We implement minibatch discrimination as a layer producing features that quantify how each sample differs 
from others in the batch:
"""
class MinibatchDiscrimination(nn.Module):
    def __init__(self, in_features, out_features, kernel_dim):
        super(MinibatchDiscrimination, self).__init__()
        # Weight tensor for projecting features (in_features -> kernel_dim) for each of out_features "kernels"
        self.T = nn.Parameter(torch.randn(in_features, out_features, kernel_dim))
        # The `T` tensor shape: [in_features, out_features, kernel_dim]
    
    def forward(self, x):
        # x shape: [batch_size, in_features]
        B = x.size(0)
        # Project input features using tensor T: result M shape [batch, out_features, kernel_dim]
        M = torch.matmul(x, self.T.view(x.size(1), -1))  # [batch, out_features*kernel_dim]
        M = M.view(B, -1, self.T.size(2))  # reshape to [batch, out_features, kernel_dim]
        # Compute pairwise distances between each pair of samples for each kernel
        # Using L1 norm for distance (could also use L2)
        diff = M.unsqueeze(0) - M.unsqueeze(1)          # shape [batch, batch, out_features, kernel_dim]
        abs_diff = torch.abs(diff).sum(dim=3)           # sum over kernel_dim -> [batch, batch, out_features]
        # Apply negative exponential to convert distances to affinity
        minibatch_features = torch.exp(-abs_diff)       # [batch, batch, out_features]
        # For each sample i, sum over j (difference from all other samples)
        minibatch_features = minibatch_features.sum(dim=1) - 1  # [batch, out_features], subtract self-distance
        return minibatch_features

class Discriminator(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Discriminator, self).__init__()
        # Simple feedforward network for example
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        # Minibatch discrimination layer (produces m extra features per sample)
        self.mb_disc = MinibatchDiscrimination(in_features=hidden_dim, out_features=50, kernel_dim=10)
        # Output layer: binary classification (real vs fake)
        self.fc_out = nn.Linear(hidden_dim + 50, 1)
    
    def forward(self, encoded_input, generated_output):
        """
        encoded_input: conditioning input embedding (e.g., from encoder)
        generated_output: candidate paraphrase (as vector or features)
        """
        # Combine input conditioning and output (for conditional GAN, e.g., concat or dot-product features)
        # Here, assume we form a joint representation by concatenation for simplicity:
        x = torch.cat([encoded_input, generated_output], dim=1) 
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        # Compute minibatch discrimination features
        mb_feats = self.mb_disc(h)
        # Concatenate minibatch features with network features
        h_cat = torch.cat([h, mb_feats], dim=1)
        # Final output (discriminator score for real/fake)
        score = self.fc_out(h_cat)
        return score

"""
In `Discriminator`, after some intermediate layers (`fc1`, `fc2`), we pass features through `MinibatchDiscrimination`. 
This returns a vector for each sample that measures how **distinct** it is from others in the batch. We concatenate 
these features to `h` and feed to `fc_out` for the final real/fake score. With this, if the generator collapses to 
identical outputs, the discriminator can spot very low distance (high similarity) among batch samples and label them 
as fake more decisively ([](http://arxiv.org/pdf/1606.03498#:~:text=what%20we%20call%20minibatch%20discrimination,than%20in%20isolation%2C%20could%20potentially)), thereby pushing the generator towards diversity.

### 3. InfoGAN-Style Latent Code Regularization

We integrate ideas from *InfoGAN* ([InfoGAN: Interpretable Representation Learning by Information Maximizing Generative Adversarial Nets](https://proceedings.neurips.cc/paper/6399-infogan-interpretable-representation-learning-by-information-maximizing-generative-adversarial-nets.pdf#:~:text=In%20this%20paper%2C%20we%20present,discover%20highly%20semantic%20and%20meaningful)) ([InfoGAN: Interpretable Representation Learning by Information Maximizing Generative Adversarial Nets](https://proceedings.neurips.cc/paper/6399-infogan-interpretable-representation-learning-by-information-maximizing-generative-adversarial-nets.pdf#:~:text=,both%20discrete%20and%20continuous%20latent)) to introduce semantically meaningful latent codes. In addition 
to the usual random noise `z`, the generator will take **discrete codes** (e.g., a one-hot vector) and **continuous 
codes** that we aim to correlate with interpretable variations in the paraphrase (such as sentence style or tone). We 
add an auxiliary loss maximizing the mutual information between these latent codes and the generated output ([InfoGAN: Interpretable Representation Learning by Information Maximizing Generative Adversarial Nets](https://proceedings.neurips.cc/paper/6399-infogan-interpretable-representation-learning-by-information-maximizing-generative-adversarial-nets.pdf#:~:text=In%20this%20paper%2C%20we%20present,discover%20highly%20semantic%20and%20meaningful)). 
This helps ensure that changing the latent code yields a perceptible change in the paraphrase, increasing **controllable 
diversity**.

We implement InfoGAN regularization by adding a small network (`Q` head) to the discriminator to predict the latent 
codes from generated samples. The generator is trained to **fool the discriminator and also to encode the latent 
code in its output**, while the `Q` network tries to correctly predict the code (maximizing mutual information).

Below, we augment the `Discriminator` with a `Q` network for code prediction and define the InfoGAN loss components:
"""
class InfoGAN_Discriminator(nn.Module):
    def __init__(self, input_dim, hidden_dim, disc_code_dim, cont_code_dim):
        super(InfoGAN_Discriminator, self).__init__()
        # Shared discriminator layers (same as before)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.mb_disc = MinibatchDiscrimination(in_features=hidden_dim, out_features=50, kernel_dim=5)
        self.fc_out = nn.Linear(hidden_dim + 50, 1)  # GAN discriminator output
        # Q-network to predict latent codes: takes the hidden features and outputs code predictions
        # For a discrete code of dimension `disc_code_dim` (treated as classification) 
        # and continuous code of dimension `cont_code_dim` (regression).
        self.q_disc = nn.Linear(hidden_dim, disc_code_dim)        # logits for discrete code
        self.q_cont_mu = nn.Linear(hidden_dim, cont_code_dim)     # predicted mean for continuous code
        # (We could predict variance for continuous as well, but assume fixed variance for simplicity)
    
    def forward(self, encoded_input, generated_output):
        # Similar to previous Discriminator forward pass
        x = torch.cat([encoded_input, generated_output], dim=1)
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        # Apply minibatch discrimination to encourage diversity
        mb_feats = self.mb_disc(h)
        h_cat = torch.cat([h, mb_feats], dim=1)
        # Discriminator output for real/fake
        score = self.fc_out(h_cat)
        # Q-network outputs for latent code prediction (use `h` before concatenation as code is global)
        disc_logits = self.q_disc(h)            # logits for discrete code categories
        cont_mu = self.q_cont_mu(h)            # predicted continuous code (mean)
        return score, disc_logits, cont_mu

# Loss functions for InfoGAN
def infogan_loss(disc_logits, true_disc_code, cont_mu, true_cont_code):
    # Discrete code loss: cross-entropy between true latent category and predicted
    disc_loss = F.cross_entropy(disc_logits, true_disc_code)
    # Continuous code loss: mean squared error (or L2) between true and predicted continuous codes
    cont_loss = F.mse_loss(cont_mu, true_cont_code)
    # Total Q loss
    return disc_loss + cont_loss

"""
Here, `InfoGAN_Discriminator` outputs both the normal GAN score and the predictions for latent codes (`disc_logits` 
for discrete code, `cont_mu` for continuous code). During training, we compute an InfoGAN **Q-loss** (`infogan_loss`) 
that measures how well the predicted codes match the generator's input codes. This loss is **minimized** by training 
the Q-network (part of discriminator) and simultaneously **maximized** w.r.t. the generator (so the generator learns 
to encode the latent info in its output) ([InfoGAN: Interpretable Representation Learning by Information Maximizing Generative Adversarial Nets](https://proceedings.neurips.cc/paper/6399-infogan-interpretable-representation-learning-by-information-maximizing-generative-adversarial-nets.pdf#:~:text=In%20this%20paper%2C%20we%20present,discover%20highly%20semantic%20and%20meaningful)). By doing so, the generator learns to produce paraphrases that 
vary according to the latent codes, yielding interpretable diversity. *InfoGAN can disentangle both discrete and continuous 
latent factors* ([InfoGAN: Interpretable Representation Learning by Information Maximizing Generative Adversarial Nets](https://proceedings.neurips.cc/paper/6399-infogan-interpretable-representation-learning-by-information-maximizing-generative-adversarial-nets.pdf#:~:text=,both%20discrete%20and%20continuous%20latent)), which in our context means we might discover latent dimensions for formality, sentence length, etc.

**Expected impact:** more **controlled diversity** – one can tweak latent codes to get different paraphrase styles or nuances, 
and the model avoids always choosing the same paraphrasing strategy.

### 4. VAE-Inspired Encoder for Input Sentences

We replace the simple skip-thought encoder with a **Variational Autoencoder (VAE)**-style encoder to obtain a robust latent 
representation of the input sentence. The encoder will produce a mean and variance for a latent Gaussian distribution, from 
which we sample a latent vector `z_enc`. This stochastic embedding encourages the model to cover variations in phrasing and 
not rely on a fixed deterministic embedding. The VAE's KL-divergence loss regularizes the latent space to follow a Gaussian 
prior ([Building and Understanding Variational Autoencoders (VAEs): A Simple MNIST Example | by Kunal Jindal | Medium](https://medium.com/@kjkjindal/building-and-understanding-variational-autoencoders-vaes-a-simple-mnist-example-9f3a774b5153#:~:text=,prior%2C%20helping%20generate%20smooth%20outputs)), promoting smoother interpolation between different inputs and potentially helping the generator to produce 
varied yet coherent outputs.

The VAE encoder structure and forward pass with reparameterization:
"""
class VAEEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(VAEEncoder, self).__init__()
        # Simple encoder network (could be an LSTM or Transformer encoder for sentences)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc_mean = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
    
    def forward(self, input_features):
        # Encode input to hidden representation
        h = F.relu(self.fc1(input_features))
        # Get parameters of latent Gaussian
        mu = self.fc_mean(h)
        logvar = self.fc_logvar(h)
        # Reparameterization trick: sample latent z from N(mu, sigma^2)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std    # sampled latent vector
        return z, mu, logvar

# Example usage:
encoder = VAEEncoder(input_dim=300, hidden_dim=200, latent_dim=100)
sample_input = torch.randn(32, 300)  # e.g., 32-dim batch of sentence embeddings
z_enc, mu, logvar = encoder(sample_input)
"""
In this `VAEEncoder`, we use a simple feed-forward network for illustration (in practice, use a Transformer/LSTM to get 
sentence features). It outputs `mu` and `logvar` for a latent vector. We then sample `z` via the reparameterization trick. 
During training, we will add a **KL-divergence** term: `KL = -0.5 * sum(1 + logvar - mu^2 - exp(logvar))` (per dimension) 
which is *minimized* to push `q(z|input)` closer to a unit Gaussian prior ([Building and Understanding Variational Autoencoders (VAEs): A Simple MNIST Example | by Kunal Jindal | Medium](https://medium.com/@kjkjindal/building-and-understanding-variational-autoencoders-vaes-a-simple-mnist-example-9f3a774b5153#:~:text=,prior%2C%20helping%20generate%20smooth%20outputs)). This ensures the encoder doesn't 
overfit to inputs and spreads out the latent space. The **expected impact** is a more **smooth and generalizable latent 
space**: similar inputs map to nearby latent vectors, aiding the generator in producing variations without deviating 
from meaning. It also provides **regularization** that can improve output quality by not over-relying on exact input encoding.

The generator will take this `z_enc` (possibly concatenated with the InfoGAN latent codes and/or noise) as input instead 
of a fixed embedding. This makes the system a hybrid VAE-GAN, leveraging both KL regularization and adversarial training.

## Training Enhancements

### 1. Curriculum Learning for Conditional Input

We apply a curriculum learning strategy to ease the generator's training. Early on, we provide the generator with the 
**ground-truth sentence embedding** (the real encoded input) frequently, so it has strong guidance for paraphrasing. 
Over time, we **gradually reduce** this help, forcing the generator to rely more on its own predicted latent (`z_enc` 
and noise) and less on the actual embedding. By the end, the generator effectively works from the latent alone (or a 
small mix), which means it learns to paraphrase with minimal direct copying.

This can be implemented by linearly annealing a probability `p_real_embed` from 1 to 0:
"""
import random

# pseudo-code for curriculum in training loop
initial_real_embed_prob = 1.0
final_real_embed_prob = 0.0
num_epochs = 100

for epoch in range(num_epochs):
    # linearly interpolate probability over epochs
    p_real_embed = initial_real_embed_prob * (1 - epoch/num_epochs) + final_real_embed_prob * (epoch/num_epochs)
    for batch in train_loader:
        input_sentence = batch["sentence"]            # original input sentence
        real_embedding = encoder(input_sentence)      # encode input via VAEEncoder
        random_noise = torch.randn_like(real_embedding)  # some random noise of same size
        # Decide whether to use real embedding or sampled latent as generator input
        if random.random() < p_real_embed:
            z_for_generator = real_embedding
        else:
            z_for_generator = random_noise  # (or could use a different sampled latent or partially use real)
        # Generate paraphrase using current generator
        paraphrase_logits = generator(z_for_generator, target_seq=None)  # target_seq None for free-running generation
        # ... compute losses, update generator and discriminator ...
"""
In the above pseudo-code, at the start `p_real_embed ~ 1.0`, so mostly we feed the actual `real_embedding` of the input 
sentence to the generator (making the task easier and ensuring correct content). As training progresses, `p_real_embed` 
decreases, and the generator more often gets only noise or its own learned latent (`z_for_generator`). This curriculum 
learning approach gradually increases task difficulty ([[PDF] Genet: Automatic Curriculum Generation for Learning Adaptation in ...](https://fyy.cs.illinois.edu/documents/genet-sigcomm22.pdf#:~:text=At%20a%20high%20level%2C%20curriculum,However)), preventing the generator from becoming overly reliant 
on the encoded input and encouraging it to learn to **generate coherent paraphrases from its latent space**. The expected 
impact is **smoother training** and improved generator autonomy, ultimately yielding a model that can paraphrase reliably 
even without teacher forcing.

### 2. Wasserstein GAN with Gradient Penalty (WGAN-GP)

We adopt the Wasserstein GAN framework with a gradient penalty for improved training stability ([[1704.00028] Improved Training of Wasserstein GANs - arXiv](https://arxiv.org/abs/1704.00028#:~:text=We%20propose%20an%20alternative%20to,Our%20proposed%20method)). WGAN treats the 
discriminator as a *critic* outputting a score instead of a probability, and the generator is trained to maximize this score 
for fake data. The **gradient penalty (GP)** term enforces the Lipschitz constraint by penalizing the norm of the critic's 
gradient w.r.t. its input ([[1704.00028] Improved Training of Wasserstein GANs - arXiv](https://arxiv.org/abs/1704.00028#:~:text=We%20propose%20an%20alternative%20to,Our%20proposed%20method)). This avoids issues caused by weight clipping in original WGAN and leads to **more stable 
and convergent training** ([[1704.00028] Improved Training of Wasserstein GANs - arXiv](https://arxiv.org/abs/1704.00028#:~:text=We%20propose%20an%20alternative%20to,Our%20proposed%20method)). In practice, WGAN-GP helps reduce mode collapse and improves output quality.

**Loss functions:** 
- Discriminator (critic) loss = *D(fake) - D(real) + λ * GP*, where GP is the gradient penalty.
- Generator loss = *-D(fake)* (trying to increase critic's score on fake outputs).

Below is an example training step with WGAN-GP, including gradient penalty computation:
"""
from torch import autograd

def compute_gradient_penalty(critic, real_data, fake_data, conditioning):
    alpha = torch.rand(real_data.size(0), 1).to(real_data.device)  # random interpolation factor
    # Interpolate between real and fake data
    interpolates = alpha * real_data + (1 - alpha) * fake_data
    interpolates.requires_grad_(True)
    # Compute critic score on interpolated data
    mixed_score = critic(conditioning, interpolates)
    # Compute gradients of score w.r.t. interpolated data
    grad_outputs = torch.ones_like(mixed_score)
    gradients = autograd.grad(
        outputs=mixed_score, inputs=interpolates,
        grad_outputs=grad_outputs,
        create_graph=True, retain_graph=True, only_inputs=True
    )[0]
    # Gradients shape: [batch_size, data_dim]; compute L2 norm per sample
    grad_norm = gradients.view(gradients.size(0), -1).norm(2, dim=1)
    # GP term: squared distance from norm=1
    gradient_penalty = ((grad_norm - 1) ** 2).mean()
    return gradient_penalty

# In training loop for each batch:
real_inputs = encoder(batch["input_sentence"])      # encode real input
real_paraphrase = batch["target_sentence_embedding"]  # or actual word seq, here assume embedding
fake_paraphrase = generator(encoder(batch["input_sentence"])[0], target_seq=None)  # generate fake paraphrase embedding
# Critic scores:
real_score = discriminator(real_inputs, real_paraphrase)
fake_score = discriminator(real_inputs, fake_paraphrase)
# Gradient penalty:
gp = compute_gradient_penalty(discriminator, real_paraphrase, fake_paraphrase, real_inputs)
# WGAN-GP losses:
critic_loss = fake_score.mean() - real_score.mean() + 10 * gp   # lambda=10
gen_loss = -fake_score.mean()
"""
In this snippet, `compute_gradient_penalty` creates an interpolated sample between a real paraphrase and a generated paraphrase, 
computes the discriminator's gradient on it, and penalizes deviation of its norm from 1 ([[1704.00028] Improved Training of Wasserstein GANs - arXiv](https://arxiv.org/abs/1704.00028#:~:text=We%20propose%20an%20alternative%20to,Our%20proposed%20method)). The constant 10 is a common 
choice for λ. We update the discriminator to minimize `critic_loss` and the generator to minimize `gen_loss`. Using WGAN-GP 
**stabilizes GAN training** and often improves the **variety and realism** of generated outputs ([[1704.00028] Improved Training of Wasserstein GANs - arXiv](https://arxiv.org/abs/1704.00028#:~:text=We%20propose%20an%20alternative%20to,Our%20proposed%20method)), which in our case 
means more natural and varied paraphrases.

### 3. KL-Divergence Regularization (VAE Loss Component)

Along with adversarial losses, we incorporate the **KL-divergence loss** from the VAE encoder into the training objective. 
This is the term that pushes the encoder's latent distribution toward the prior N(0, I). By adding `loss_KL = β * KL(mu, logvar)` 
to the generator/encoder loss (with β as a weight, often 1.0), we ensure the encoder's latent space remains smooth and prevents 
overfitting to the training data. This should improve the **generalization** to new inputs and maintain diversity in generated 
paraphrases. As noted earlier, the KL loss encourages latent space to follow a Gaussian prior ([Building and Understanding Variational Autoencoders (VAEs): A Simple MNIST Example | by Kunal Jindal | Medium](https://medium.com/@kjkjindal/building-and-understanding-variational-autoencoders-vaes-a-simple-mnist-example-9f3a774b5153#:~:text=,prior%2C%20helping%20generate%20smooth%20outputs)), which helps in 
generating *smooth* output variations.

In code, after obtaining `mu` and `logvar` from the `VAEEncoder` for an input, we compute:
```python
KL_loss = -0.5 * torch.mean(1 + logvar - mu**2 - logvar.exp())
total_gen_loss = gen_loss + KL_loss_weight * KL_loss + info_loss_weight * InfoGAN_loss
```
Here `KL_loss_weight` might be annealed via a schedule (KL *warm-up* is common: start low, then increase weight) to avoid 
over-regularizing at the very beginning. The **expected impact** is a latent space that better covers possible paraphrase 
variations, improving **diversity** without sacrificing grammaticality.

### 4. Mixed Precision Training

To accelerate training, we enable **mixed precision** (FP16/FP32) training. Mixed precision uses half-precision for most 
operations while keeping a few in full precision to preserve numerical stability ([Train With Mixed Precision - NVIDIA Docs](https://docs.nvidia.com/deeplearning/performance/mixed-precision-training/index.html#:~:text=Mixed%20precision%20training%20offers%20significant,format%2C%20while%20storing%20minimal)). This can significantly speed up 
training on modern GPUs and reduce memory usage ([Train With Mixed Precision - NVIDIA Docs](https://docs.nvidia.com/deeplearning/performance/mixed-precision-training/index.html#:~:text=Mixed%20precision%20training%20offers%20significant,format%2C%20while%20storing%20minimal)), allowing larger batch sizes or models. We expect **faster iterations** 
and the ability to train more epochs or try more hyperparameters in the same time budget, indirectly leading to a better model.

Implementation involves using frameworks like PyTorch's Automatic Mixed Precision (AMP):
```python
scaler = torch.cuda.amp.GradScaler()
for batch in train_loader:
    optimizer_G.zero_grad(); optimizer_D.zero_grad()
    with torch.cuda.amp.autocast():
        # forward passes
        z_enc, mu, logvar = encoder(batch["input"])
        fake = generator(z_enc, target_seq=None)
        real = batch["target_embedding"]
        real_score = discriminator(z_enc, real)
        fake_score = discriminator(z_enc, fake)
        # compute losses (WGAN-GP, KL, InfoGAN, etc.)
        gp = compute_gradient_penalty(discriminator, real, fake, z_enc)
        D_loss = fake_score.mean() - real_score.mean() + 10 * gp
        G_adv_loss = -fake_score.mean()
        KL_loss = -0.5 * torch.mean(1 + logvar - mu**2 - logvar.exp())
        # assume disc_code and cont_code predicted from discriminator for InfoGAN:
        _, disc_logits, cont_pred = discriminator(z_enc, fake)
        info_loss = infogan_loss(disc_logits, batch["disc_code"], cont_pred, batch["cont_code"])
        G_loss = G_adv_loss + KL_loss + info_loss
    # backprop with scaled loss
    scaler.scale(D_loss).backward(retain_graph=True)
    scaler.step(optimizer_D)
    scaler.update()
    scaler.scale(G_loss).backward()
    scaler.step(optimizer_G)
    scaler.update()
```
With `torch.cuda.amp.autocast()`, operations inside run in float16 where safe. The `GradScaler` handles scaling the loss 
to avoid underflow, then un-scaling before `step`. Mixed precision typically yields **1.5x to 3x training speedups** without 
loss of accuracy ([Automatic Mixed Precision for Deep Learning | NVIDIA Developer](https://developer.nvidia.com/automatic-mixed-precision#:~:text=Developer%20developer,On)). Faster training means we can iterate more and potentially find a better model.

## Evaluation Metrics

We expand our evaluation to multiple metrics to get a comprehensive view of quality and diversity:

- **BLEU-1 to BLEU-4**: Measures n-gram overlap between generated paraphrase and reference. BLEU is the classic MT metric ([Jointly Measuring Diversity and Quality in Text Generation Models](https://aclanthology.org/W19-2311.pdf#:~:text=BLEU%3A%20It%20is%20the%20most,like%20machine%20translation%20which%20include)); 
  BLEU-4 (up to 4-grams) is especially common for full sentence similarity. We will report BLEU-1,2,3,4 to see how well the model 
  preserves meaning (higher = more overlap with reference).

- **METEOR**: A metric that considers synonyms and stemming in addition to precision/recall of unigrams ([METEOR - Wikipedia](https://en.wikipedia.org/wiki/METEOR#:~:text=METEOR%20,and%20recall%2C%20with%20recall)). METEOR often 
  correlates better with human judgment for paraphrase and translation than BLEU, since it accounts for *paraphrasing* (e.g., matching 
  "small" with "little") ([Meteor++ 2.0: Adopt Syntactic Level Paraphrase Knowledge into ...](https://aclanthology.org/W19-5357/#:~:text=,focuses%20on%20the%20lexical%20level)). This will better reflect semantic adequacy.

- **ROUGE-L**: Measures recall of the longest common subsequence (LCS) between output and reference ([Two minutes NLP — Learn the ROUGE metric by examples - Medium](https://medium.com/nlplanet/two-minutes-nlp-learn-the-rouge-metric-by-examples-f179cc285499#:~:text=ROUGE,longest%20sequence%20of%20words)). ROUGE-L focuses on how 
  much of the reference content is covered in the output (important for paraphrase which should retain meaning). A high ROUGE-L means 
  the paraphrase contains a long sequence of words in common with the reference, indicating strong overlap in content and structure.

- **Self-BLEU**: We introduce Self-BLEU ([Jointly Measuring Diversity and Quality in Text Generation Models](https://aclanthology.org/W19-2311.pdf#:~:text=Self,By%20averaging%20these%20BLEU)) as a *diversity* metric. Self-BLEU computes the BLEU score of each generated output 
  using other generated outputs as references ([Jointly Measuring Diversity and Quality in Text Generation Models](https://aclanthology.org/W19-2311.pdf#:~:text=Self,By%20averaging%20these%20BLEU)). If the model produces very similar sentences each time, Self-BLEU will be high 
  (because each output matches others), indicating low diversity. We want **low Self-BLEU** (close to 0) meaning outputs are diverse ([Jointly Measuring Diversity and Quality in Text Generation Models](https://aclanthology.org/W19-2311.pdf#:~:text=Self,By%20averaging%20these%20BLEU)). 
  This guards against mode collapse: as diversity increases, Self-BLEU drops.

- **Distinct-1 and Distinct-2**: These metrics count the proportion of unique unigrams and bigrams in the generated outputs ([arXiv:1510.03055v2  [cs.CL]  7 Jan 2016](https://nlp.stanford.edu/pubs/jiwei2016diversity.pdf#:~:text=models.%20distinct,2)). 
  For example, Distinct-1 is the number of distinct words divided by total words across all outputs ([arXiv:1510.03055v2  [cs.CL]  7 Jan 2016](https://nlp.stanford.edu/pubs/jiwei2016diversity.pdf#:~:text=models.%20distinct,2)). Higher distinct-1/2 means 
  lexically diverse outputs (using varied vocabulary and word combinations). We expect that adding InfoGAN, minibatch discrimination, etc., 
  will increase these distinctness metrics. Distinct-N was introduced to evaluate diversity in dialog generation ([arXiv:1510.03055v2  [cs.CL]  7 Jan 2016](https://nlp.stanford.edu/pubs/jiwei2016diversity.pdf#:~:text=models.%20distinct,2)), and here it 
  will help quantify paraphrase diversity (we want the model not to always say e.g. "I have X" for every input).

By tracking BLEU and ROUGE-L, we ensure the paraphrases are faithful to the input meaning. METEOR gives a sense of semantic adequacy with 
paraphrasing in mind. Then Self-BLEU and Distinct-N specifically address the **diversity** of model outputs, to ensure we didn't trade off 
quality for mode collapse. These metrics together provide a balanced evaluation of **quality vs. diversity**.

In code, we can compute these with libraries (e.g., NLTK or SacreBLEU for BLEU, NLTK or `datasets` for METEOR and ROUGE, and custom 
functions for Self-BLEU and Distinct). For instance, computing distinct metrics:
```python
def distinct_n(sentences, n):
    # sentences: list of generated sentences (each a list of tokens)
    ngrams = set()
    total_ngrams = 0
    for tokens in sentences:
        grams = list(zip(*[tokens[i:] for i in range(n)]))
        total_ngrams += len(grams)
        ngrams.update(grams)
    if total_ngrams == 0:
        return 0.0
    return len(ngrams) / total_ngrams

distinct_1 = distinct_n(generated_sentences, 1)
distinct_2 = distinct_n(generated_sentences, 2)
```
We would report e.g. `BLEU-4: X, METEOR: Y, ROUGE-L: Z, Self-BLEU: A, Distinct-1/2: B/C` to summarize performance. 

## Decoding Enhancements

Beyond greedy decoding, we implement **advanced sampling strategies** to improve the generated paraphrase quality:

- **Top-k Sampling**: Limit the next-word sampling to the top k most probable tokens (e.g., k=50) from the model's distribution. This avoids 
  extremely unlikely words. It's a way to inject randomness while preventing outlandish choices. Top-k picks from a fixed-size most likely set ([Top-p sampling - Wikipedia](https://en.wikipedia.org/wiki/Top-p_sampling#:~:text=Top,3)).

- **Nucleus (Top-p) Sampling**: Instead of a fixed k, choose the smallest set of top tokens whose cumulative probability exceeds p (e.g., p=0.9) ([Top-p sampling - Wikipedia](https://en.wikipedia.org/wiki/Top-p_sampling#:~:text=that%20are%20repetitive%20and%20otherwise,rest%20of%20tokens%20are%20rejected)). 
  This **dynamic** cutoff (the "nucleus") adapts to the uncertainty of the distribution ([Top-p sampling - Wikipedia](https://en.wikipedia.org/wiki/Top-p_sampling#:~:text=that%20are%20repetitive%20and%20otherwise,rest%20of%20tokens%20are%20rejected)). It generally yields more natural outputs by including 
  enough options to cover ~90% probability mass, cutting off the long tail of very unlikely words that can cause incoherence ([The Curious Case of Neural Text Degeneration - OpenReview](https://openreview.net/forum?id=rygGQyrFvH#:~:text=The%20Curious%20Case%20of%20Neural,from%20the%20dynamic%20nucleus)). Top-p is often preferred 
  for open-ended generation as it balances coherence and creativity.

- **Temperature Scaling**: We allow an adjustable `temperature` parameter in sampling. This scales the logits before softmax: higher T > 1 makes the distribution 
  flatter (more random output), while lower T < 1 makes it peakier (more greedy/deterministic). This gives control over output creativity vs. determinism ([The Art of Sampling: Controlling Randomness in LLMs](https://www.anup.io/p/the-art-of-sampling-controlling-randomness#:~:text=High%20temperature%20%28T%20,Sharpens%20the)). 
  For example, T=1 is normal, T=0.7 produces more conservative choices (less randomness), T=1.5 produces more varied, sometimes surprising paraphrases. We will tune 
  T to see what yields best quality/diversity trade-off.

- **Beam Search and Length Penalty**: For completeness, we can also use beam search with a small beam width (e.g., 5) to find high-probability paraphrases. Beam search 
  tends to maximize BLEU but can reduce diversity (often converging to safe, common phrasings). We will experiment with modest beam sizes to avoid dull outputs. We can 
  apply a length penalty to the beam scoring to prevent the beam search from favoring very short outputs. This ensures paraphrases are of reasonable length and completeness.

**Implementing sampling:** We create utilities for top-k and top-p:
"""
import torch.nn.functional as F

def sample_with_topk_topp(logits, top_k=0, top_p=0.0, temperature=1.0):
    """Apply top-k and/or nucleus (top-p) filtering to logits and sample a token."""
    # Apply temperature
    logits = logits / temperature
    probs = F.softmax(logits, dim=-1)
    # If top-p is set, use nucleus sampling
    if top_p > 0.0:
        # Sort probabilities descending
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        # Mask out tokens beyond the top-p nucleus
        cutoff = (cumulative_probs > top_p).nonzero(as_tuple=True)[0]
        if len(cutoff) > 0:
            cutoff_index = cutoff[0]
            sorted_probs = sorted_probs[:cutoff_index+1]
            sorted_indices = sorted_indices[:cutoff_index+1]
        else:
            cutoff_index = sorted_probs.size(0)
        # Re-normalize within the top-p set
        probs = torch.zeros_like(probs)
        probs[sorted_indices] = sorted_probs / sorted_probs.sum()
    # If top-k is set (and top-p was not used or remaining set still large)
    if top_k > 0:
        topk_vals, topk_idx = torch.topk(probs, top_k)
        probs = torch.zeros_like(probs)
        probs[topk_idx] = topk_vals
        probs = probs / probs.sum()
    # Sample from the final distribution
    token = torch.multinomial(probs, num_samples=1)
    return token.item()

# Example decoding loop using the above sampler:
def generate_paraphrase(model, input_embedding, max_len=30, top_k=50, top_p=0.9, temperature=1.0):
    generated_tokens = []
    for t in range(max_len):
        logits_t = model(input_embedding, torch.tensor([generated_tokens]))[:, -1, :]
        # (Assume model returns logits for the sequence, take last timestep)
        token_id = sample_with_topk_topp(logits_t.squeeze(0), top_k=top_k, top_p=top_p, temperature=temperature)
        if token_id == EOS_token:  # EOS_token is a defined end-of-sentence token
            break
        generated_tokens.append(token_id)
    return generated_tokens
"""
This `sample_with_topk_topp` function first applies temperature scaling to logits. Then, if `top_p` is specified, it selects 
the nucleus of probable tokens (those covering probability mass >= p) ([Top-p sampling - Wikipedia](https://en.wikipedia.org/wiki/Top-p_sampling#:~:text=that%20are%20repetitive%20and%20otherwise,rest%20of%20tokens%20are%20rejected)) and redistributes probability only among them. 
If `top_k` is specified, it further restricts to the top k tokens ([Top-p sampling - Wikipedia](https://en.wikipedia.org/wiki/Top-p_sampling#:~:text=Top,3)). Finally, it samples one token from the filtered 
distribution. We can use both top_k and top_p (though typically one uses one or the other).

Using **nucleus sampling** (say p=0.9) typically yields paraphrases that are **fluent and diverse**, avoiding the common problem 
of repetitive or generic outputs that deterministic decoders (greedy/beam) might give ([The Curious Case of Neural Text Degeneration - OpenReview](https://openreview.net/forum?id=rygGQyrFvH#:~:text=The%20Curious%20Case%20of%20Neural,from%20the%20dynamic%20nucleus)). We will compare outputs from 
greedy, beam, top-k, and nucleus to choose decoding parameters that give the best balance of keeping the meaning while not 
being too conservative.

Additionally, we will tune the **beam size** and **max length**. For example, increasing the beam size might improve BLEU slightly 
but can hurt distinctness; we might settle on a beam of 5 or use nucleus sampling for final results. The max length is important to 
avoid run-on sentences; we set a reasonable limit (like 30 tokens) based on the dataset's typical sentence lengths. Early stopping 
on EOS ensures we don't overshoot.

## Project Structure and Refactoring

To manage these improvements, we refactor the project into a modular structure:

```
project/
├── **data/**              # data loading & preprocessing
│   └── dataset.py         # dataset class, data augmentation methods
├── **model/**             # model architecture components
│   ├── generator.py       # Generator (Transformer decoder with BN)
│   ├── discriminator.py   # Discriminator (with minibatch disc, Q network)
│   ├── encoder.py         # VAE encoder for input sentences
│   └── __init__.py
├── **training/**          # training loop and loss functions
│   ├── trainer.py         # adversarial training routine (GAN + VAE losses)
│   ├── losses.py          # definitions for WGAN-GP loss, InfoGAN loss, KL loss
│   └── __init__.py
├── **evaluation/**        # evaluation metrics and methods
│   ├── metrics.py         # functions to compute BLEU, METEOR, ROUGE, Self-BLEU, Distinct
│   └── __init__.py
├── **utils/**             # utility functions (e.g., decoding methods, config utils)
│   ├── decoding.py        # top-k, top-p sampling implementations
│   ├── helpers.py         # misc utilities, e.g., seed setting
│   └── __init__.py
├── **configs/**           # configuration files for experiments
│   ├── default.yaml       # default hyperparameters and settings
│   └── exp1.yaml          # example of a specific experiment config
└── **main.py**            # entry point to train or evaluate the model
```

This structure separates concerns: e.g., `model` contains only network definitions, `training` has the loop and can import 
losses and model components, `evaluation` has metrics logic, etc. This makes the code easier to navigate and test.

For instance, `model/generator.py` will contain our `Generator` class with BN, and possibly a separate `VAEDecoder` if needed. 
`model/discriminator.py` contains `InfoGAN_Discriminator` class. The `training/trainer.py` will orchestrate an epoch: 
getting data from `data/dataset.py`, feeding it to models, computing losses from `training/losses.py` (which includes our 
WGAN-GP and KL calculations), then optimizer steps.

We also implement **command-line configuration** handling. The idea is to load a config (YAML or JSON) for hyperparameters 
and model settings, and save a copy for reproducibility. In `main.py` we can use Python's `argparse` to accept a `--config` 
file path, and then load it. For example:
"""
import yaml
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Paraphrase GAN")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to config file")
    args = parser.parse_args()
    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)
    # (The config might include settings like learning rates, latent dimensions, dataset path, etc.)
    
    # Save a copy of config to output directory for record
    import os
    os.makedirs(config["output_dir"], exist_ok=True)
    with open(os.path.join(config["output_dir"], "used_config.yaml"), 'w') as f:
        yaml.safe_dump(config, f)
    
    # Initialize dataset, model, optimizer based on config
    # e.g., dataset = ParaphraseDataset(config["data_path"], split="train")
    #       generator = Generator(**config["generator_params"])
    #       discriminator = InfoGAN_Discriminator(**config["discriminator_params"])
    # ...
    # Trainer loop etc.
```
We use `yaml.safe_load` to parse the YAML config. After training completes (or even at startup), we dump the exact config to 
`used_config.yaml` in the results folder for reproducibility. This practice ensures that each experiment's settings are saved. 
The configuration can contain all our knobs: e.g., `KL_loss_weight`, `info_loss_weight`, `batch_size`, `learning_rate`, 
`top_k`, `top_p`, etc., making experiments easily adjustable without hard-coding values.

Finally, we ensure our `README.md` and documentation reflect the new structure and how to run training/evaluation with different configs.

## Dataset and Validation

Currently, the system likely trains on a given paraphrase dataset. We will maintain that for continuity, but also consider **augmenting** 
or switching to stronger paraphrase datasets:

- **Quora Question Pairs (QQP)**: A popular dataset of over 400k question pairs with a binary label if they are paraphrases ([Quora Question Pairs Dataset | Papers With Code](https://paperswithcode.com/dataset/quora-question-pairs#:~:text=,for)). 
  We can use the paraphrase-labeled pairs as training data (question1 -> question2 as a paraphrase pair). This dataset covers a broad range 
  of everyday topics and will teach the model many ways to ask the same question. 

- **PAWS (Paraphrase Adversaries from Word Scrambling)** ([PAWS: Paraphrase Adversaries from Word Scrambling - ACL Anthology](https://aclanthology.org/N19-1131/#:~:text=Existing%20paraphrase%20identification%20datasets%20lack,accuracy%29%3B%20however)): Focuses on challenging paraphrase cases with high lexical overlap but 
  differing meaning. PAWS has 108k pairs with careful human annotation ([PAWS: Paraphrase Adversaries from Word Scrambling - ACL Anthology](https://aclanthology.org/N19-1131/#:~:text=Existing%20paraphrase%20identification%20datasets%20lack,accuracy%29%3B%20however)). Training on PAWS can help the model not to be tricked by 
  superficial word overlap and to truly understand sentence structure differences. It might reduce unrealistic paraphrases that are just 
  word swaps.

- **ParaNMT-50M** ([ParaNMT-50M: Pushing the Limits of Paraphrastic Sentence Embeddings with Millions of Machine Translations - ACL Anthology](https://aclanthology.org/P18-1042/#:~:text=We%20describe%20ParaNMT,it%20can%20be%20used%20for)): A massive dataset of 50 million paraphrase pairs generated via back-translation ([ParaNMT-50M: Pushing the Limits of Paraphrastic Sentence Embeddings with Millions of Machine Translations - ACL Anthology](https://aclanthology.org/P18-1042/#:~:text=We%20describe%20ParaNMT,it%20can%20be%20used%20for)). While automatically 
  generated, its sheer size can provide the model with a diverse range of paraphrase patterns. We could use a subset (e.g., 5M high-quality 
  pairs) due to resource constraints. Pretraining on ParaNMT and fine-tuning on a smaller human-curated set could combine coverage with accuracy.

Each of these datasets can be explored for improved performance. Initially, we keep the current dataset to measure progress fairly, but 
these are promising for future improvement. We should preprocess these (tokenization, maybe filtering by length).

Additionally, we introduce a **validation split** from the training data (or use provided dev sets like Quora's) to perform early stopping 
and hyperparameter tuning. For example, we can set aside 5-10% of training pairs as `validation`. During training, after each epoch (or every N 
iterations), we compute validation BLEU/METEOR etc. If the validation metrics stop improving (or if a combined score plateaus) for several epochs, 
we stop training to prevent overfitting. Early stopping ensures we **don't over-train** the generator to just mimic training pairs, which can 
hurt generalization to new inputs.

In code, this might look like:
```python
best_val_score = -float("inf")
epochs_no_improve = 0
for epoch in range(max_epochs):
    train_epoch(...)  # perform training on full epoch
    val_scores = evaluate_model(generator, val_loader)  # compute BLEU, etc. on validation set
    if val_scores["BLEU-4"] + val_scores["METEOR"] > best_val_score:  # example criterion
        best_val_score = val_scores["BLEU-4"] + val_scores["METEOR"]
        epochs_no_improve = 0
        # save model checkpoint
        torch.save(generator.state_dict(), "best_generator.pth")
        torch.save(discriminator.state_dict(), "best_discriminator.pth")
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= patience:
            print("Early stopping triggered.")
            break
```
We might use a composite of BLEU-4 and METEOR (or any preferred metric) as the validation criterion. The model checkpointing ensures we keep the best model.

**Summary of Dataset improvements:** By training (or fine-tuning) on datasets like QQP, PAWS, and ParaNMT, the model will learn from a **wider 
variety of paraphrase examples**, likely improving its ability to generalize to new inputs and produce more diverse paraphrases. The validation 
split and early stopping will help us use these datasets effectively without overfitting, leading to **better generalization and reliability** 
when generating paraphrases on unseen data.

## Conclusion and Expected Impact

By implementing the above enhancements, we transform the paraphrase GAN into a more powerful system:

- **Architecture**: A deeper, normalized generator for quality; discriminator with minibatch and InfoGAN components for diversity and controllability; 
  VAE encoder for a smooth latent space. *Impact:* more fluent outputs and better diversity with interpretable variation.

- **Training**: Curriculum learning and WGAN-GP for stable GAN convergence; KL regularization to maintain latent space structure; mixed precision for speed. 
  *Impact:* faster training, more stable adversarial updates (less collapse), model learns a rich latent representation while preserving input meaning.

- **Decoding**: Advanced sampling (top-k, nucleus, temperature) and beam tuning. *Impact:* ability to trade-off diversity and correctness at generation time, 
  avoiding dull or degenerate outputs and yielding more natural paraphrases.

- **Evaluation**: A broad suite of metrics (BLEU, METEOR, ROUGE-L for quality; Self-BLEU, Distinct for diversity) to properly measure progress. 
  *Impact:* better insight into model performance, ensuring improvements in one aspect (e.g., diversity) don’t secretly harm another (e.g., quality).

- **Project structure**: Modular code and config-driven experiments. *Impact:* easier maintenance and iteration, making it straightforward to test new ideas 
  (like using a new dataset or adjusting a loss weight) and reproduce results.

- **Dataset**: Access to large and challenging paraphrase corpora and using validation for tuning. *Impact:* model learns from the best available data and 
  we safeguard against overfitting, leading to a more robust paraphrase generator.

With these enhancements, we expect the GAN-based paraphrase system to generate sentences that are not only close paraphrases of the input (high BLEU/ROUGE, 
meaning preserved) but also varied in wording and structure (high distinct-N, low self-BLEU), all while training more stably. This lays a strong foundation 
for high-quality paraphrase generation without relying on enormous pre-trained models, achieving a good balance between **linguistic quality** and **output diversity**.
"""
```