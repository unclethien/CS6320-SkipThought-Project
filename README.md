### CS6320 Natural Language Processing

### Transformer-Based Latent GAN for Authorial Text Generation

<!-- Task -->

## Task

Due date: Fri, Apr 18

- [ ] Skip Through Vector Rebuild as soon as we can.
- [ ] Transformer-Based GAN Alternative

<!-- Run this project -->

## How to Run

This project uses a modular structure and relies on PyTorch. It's recommended to run it within a dedicated virtual environment, either on Windows directly or using WSL (Windows Subsystem for Linux).

### 1. Setting Up the Environment

First, clone the repository and navigate into the project directory.

**Create a virtual environment:**

*   **Using `venv` (Recommended):**
    ```bash
    # In your project directory (Windows CMD/PowerShell or Linux/WSL terminal)
    python -m venv .venv 
    ```

**Activate the virtual environment:**

*   **Windows (CMD):**
    ```cmd
    .venv\Scripts\activate.bat
    ```
*   **Windows (PowerShell):**
    ```powershell
    .venv\Scripts\Activate.ps1
    ```
*   **Linux / macOS / WSL:**
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

**(Placeholder)** The current `data/dataset.py` contains placeholder logic. You need to:
1.  Modify `data/dataset.py` to load your specific paraphrase dataset (e.g., from text files, CSV).
2.  Ensure the dataset class yields dictionaries with the keys expected by the `Trainer` in `training/trainer.py`.
3.  Update the tokenizer loading in `main.py` if you are not using the default Hugging Face `AutoTokenizer` approach (currently placeholder).

### 3. Configuration

Training parameters, model dimensions, file paths, etc., are controlled via YAML configuration files located in the `configs/` directory.

*   Modify `configs/default.yaml` or create new configuration files as needed.
*   Key settings include `model_params`, `training_params`, `data_params`, etc.

### 4. Training the Model

Once the environment is set up and the data loading is implemented, start the training process using `main.py` and specify a configuration file:

```bash
# Ensure your virtual environment is active
python main.py --config configs/default.yaml
```

The script will:
*   Load the specified configuration.
*   Initialize the VAE Encoder, Generator, and Discriminator models.
*   Load the tokenizer and dataset.
*   Start the training loop managed by the `Trainer` class.
*   Handle device placement (CPU/GPU).
*   (Optionally) Use mixed precision training if enabled in the config.
*   Save model checkpoints to the directory specified in the config.

### 5. Evaluation

Evaluation metrics (BLEU, ROUGE, METEOR, Distinct-N, Self-BLEU) are implemented in `evaluation/metrics.py`.

*   The `Trainer` class includes logic to perform periodic evaluation during training based on the configuration.
*   To run evaluation on a saved model checkpoint against a test set, you might need to adapt `main.py` or create a separate evaluation script that loads the model and runs the metrics from `evaluation/metrics.py`.

### Project Structure

The project follows a modular structure:

-   **`main.py`**: Main entry point to start training.
-   **`configs/`**: Contains YAML configuration files (e.g., `default.yaml`).
-   **`data/`**: Data loading and processing.
    -   `dataset.py`: Defines the PyTorch Dataset and DataLoader logic (requires implementation for your data).
-   **`model/`**: Model definitions.
    -   `encoder.py`: VAE Encoder.
    -   `generator.py`: Transformer-based Generator.
    -   `discriminator.py`: InfoGAN Discriminator with Minibatch Discrimination.
-   **`training/`**: Training loop and loss functions.
    -   `trainer.py`: Manages the training process (model updates, optimization, evaluation calls).
    -   `losses.py`: WGAN-GP loss, InfoGAN loss components.
-   **`evaluation/`**: Evaluation metrics and logic.
    -   `metrics.py`: Implementation of BLEU, ROUGE, METEOR, Distinct-N, Self-BLEU.
-   **`utils/`**: Utility functions.
    -   `helpers.py`: General helper functions (e.g., `set_seed`).
    -   `decoding.py`: Sampling functions (e.g., top-k/top-p).
-   **`requirements.txt`**: Project dependencies.
-   **`checkpoints/`**: (Created during training) Directory to save model checkpoints.
-   **`results/`**: (Optional) Directory to save generated samples or evaluation outputs.

### Download NLTK Data

Required NLTK resources (`punkt`, `wordnet`) are now auto-downloaded at runtime by `main.py`. No manual download is needed.

### Running on WSL (Windows Subsystem for Linux)

If using WSL:
1.  Access your project directory via `/mnt/<drive_letter>/...`, e.g., `/mnt/d/Project/CS6320-SkipThought-Project`.
2.  Follow the environment setup steps above within your WSL terminal.
3.  Ensure file paths in configuration files are correct for the Linux environment if using absolute paths.
4.  For GPU support, ensure CUDA is correctly configured for WSL2.
