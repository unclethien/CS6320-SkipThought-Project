### CS6320 Natural Language Processing

### Transformer-Based Latent GAN for Authorial Text Generation

<!-- Task -->

## Goal

This project implements a Generative Adversarial Network (GAN) that operates in the latent space of a Transformer-based autoencoder. The primary goal is to generate coherent and diverse text samples that mimic the style and content of a given corpus (e.g., BookCorpus).

<!-- Run this project -->

## How to Run

This project uses a modular structure and relies on PyTorch. It's recommended to run it within a dedicated virtual environment, either on Windows directly or using WSL (Windows Subsystem for Linux).

### 1. Setting Up the Environment

First, clone the repository and navigate into the project directory.

**Create a virtual environment:**

- **Using `venv` (Recommended):**
  ```bash
  # In your project directory (Windows CMD/PowerShell or Linux/WSL terminal)
  python -m venv .venv
  ```

**Activate the virtual environment:**

- **Windows (CMD):**
  ```cmd
  .venv\Scripts\activate.bat
  ```
- **Windows (PowerShell):**
  ```powershell
  .venv\Scripts\Activate.ps1
  ```
- **Linux / macOS / WSL:**
  ```bash
  source .venv/bin/activate
  ```
  Your terminal prompt should now show `(.venv)` at the beginning.

**Install dependencies:**

Install all required packages using the `requirements.txt` file:

```bash
# Ensure your virtual environment is active
pip install -r requirements.txt
```

### 2. Data Preparation

The script `data/dataset.py` handles loading datasets specified in the configuration (e.g., 'bookcorpus' from the Hugging Face `datasets` library). It automatically performs train, validation, and test splits based on percentages defined in the configuration file (`configs/default.yaml`). It also supports using a smaller subset of the data for faster testing.

### 3. Configuration

Training parameters, model dimensions, file paths, etc., are controlled via YAML configuration files located in the `configs/` directory.

- Modify `configs/default.yaml` or create new configuration files as needed.
- Key settings include `model_params`, `training_params`, `data_params`, etc.

### 4. Training the Model

Once the environment is set up, start the training process using `main.py` and specify a configuration file:

```bash
# Start training from scratch
python main.py --config configs/default.yaml

# Resume training from a specific checkpoint
python main.py --config configs/default.yaml --resume_checkpoint path/to/your/checkpoint.pth.tar
```

The script will:

- Load the specified configuration.
- Initialize the VAE Encoder, Generator, and Discriminator models.
- Load the tokenizer and dataset.
- Start the training loop managed by the `Trainer` class.
- Handle device placement (CPU/GPU).
- (Optionally) Use mixed precision training if enabled in the config.
- Save model checkpoints periodically to the `results/` directory specified in the config.

### 5. Evaluation

Evaluation metrics are calculated within the `Trainer` class (`training/trainer.py`) using the Hugging Face `evaluate` library and custom functions for diversity metrics.

- **Metrics:** BLEU, METEOR, ROUGE-L, BERTScore, Distinct-N (Distinct-1, Distinct-2), and Self-BLEU.
- The `Trainer` performs periodic evaluation on the validation set during training based on the `evaluation.eval_every_epochs` setting in the configuration.
- After training completes, a final evaluation is run on the test set.
- Results are logged to the console and TensorBoard.

### Project Structure

The project follows a modular structure:

- **`main.py`**: Main entry point to start training.
- **`configs/`**: Contains YAML configuration files (e.g., `default.yaml`).
- **`data/`**: Data loading and processing.
  - `dataset.py`: Defines the PyTorch Dataset and DataLoader logic, including splitting.
- **`model/`**: Model definitions.
  - `text_encoder.py`: Transformer-based Sentence Encoder.
  - `text_decoder.py`: Transformer-based Text Decoder.
  - `gan_networks.py`: Latent space Generator (MLP) and Discriminator (MLP).
- **`training/`**: Training loop and related logic.
  - `trainer.py`: Manages the training process (model updates, optimization, evaluation, checkpointing, logging). Handles WGAN-GP loss and reconstruction loss internally.
- **`evaluation/`**: (Directory might contain evaluation scripts or analysis, but core metric calculation is in trainer.py)
- **`utils/`**: Utility functions.
  - `helpers.py`: General helper functions (e.g., `set_seed`).
  - `decoding.py`: Sampling functions (e.g., top-k/top-p).
- **`requirements.txt`**: Project dependencies.
- **`results/`**: (Created during training) Directory to save model checkpoints, TensorBoard logs, and potentially generated samples or evaluation outputs.

### Download NLTK Data

Required NLTK resources (`punkt`, `wordnet`) are now auto-downloaded at runtime by `main.py`. No manual download is needed.

### Running on WSL (Windows Subsystem for Linux)

If using WSL:

1.  Access your project directory via `/mnt/<drive_letter>/...`, e.g., `/mnt/d/Project/CS6320-SkipThought-Project`.
2.  Follow the environment setup steps above within your WSL terminal.
3.  Ensure file paths in configuration files are correct for the Linux environment if using absolute paths.
4.  For GPU support, ensure CUDA is correctly configured for WSL2.
