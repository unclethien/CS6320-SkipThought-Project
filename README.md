### CS6320 Natural Language Processing

### Transformer-Based Latent GAN for Authorial Text Generation

<!-- Task -->

## Task

Due date: Fri, Apr 18

- [ ] Skip Through Vector Rebuild as soon as we can.
- [ ] Transformer-Based GAN Alternative

<!-- Run this project -->

## How to run

### Setting Up the Environment

First, create a new virtual environment for this project:

```sh
# Create a new virtual environment
python3 -m venv skip_thought_env

# Activate the virtual environment
# On Windows:
skip_thought_env\Scripts\activate
# On Unix or MacOS:
source skip_thought_env/bin/activate
```

After activating the environment, your command prompt should show the environment name, indicating it's active and ready for package installation.

### Prerequisites

Check if you have the required libraries installed:

```sh
pip list | grep -E "torch|numpy|nltk|transformers"
```

If not, let's install them:

```sh
pip install torch transformers numpy nltk matplotlib
```

### Data Preparation

Before training, you can preprocess your data to improve training quality:

```sh
# Clean and preprocess the training data
python preprocess_data.py --input Data/train.txt --output Data/train_processed.txt --min_length 5 --max_length 40
```

This will clean the data by:
- Removing excessive punctuation
- Normalizing text formatting
- Filtering sentences by length
- Creating a validation set

### Training the Model

Navigate to the project directory and activate the virtual environment:

```sh
cd "/mnt/d/Project/CS6320-SkipThought-Project"
source skip_thought_env/bin/activate
python Second-Skip-Though-GAN.py
```

The new improved training process includes:
- Advanced text cleaning
- Better sampling strategies
- Learning rate scheduling
- Model checkpointing

### Generating Text

After training, you can generate text samples with different parameters:

```sh
# Generate text with fixed parameters
python generate_text.py --num_samples 20 --save_path results/generated_samples.json

# Generate text with varying parameters to find optimal settings
python generate_text.py --vary_params --save_path results/parameter_exploration.json
```

### Evaluating Results

To evaluate the quality of your generated text:

```sh
python evaluate_model.py --samples_path results/generated_samples.json --reference_path Data/train_processed.txt --use_gpt2
```

This will calculate various metrics including:
- BLEU scores (1-4)
- METEOR score
- Diversity metrics (distinct-n, self-BLEU)
- Perplexity (with GPT-2 option)

The evaluation results and visualization plots will be saved to the specified output directory.

### Project Structure

- `Second-Skip-Though-GAN.py` - Main GAN model with BART encoder/decoder
- `model_adapter.py` - Module adapter for importing model components
- `generate_text.py` - Text generation with configurable parameters
- `preprocess_data.py` - Data cleaning and preparation
- `evaluate_model.py` - Advanced evaluation metrics
- `Data/` - Training and validation data
- `checkpoints/` - Saved model checkpoints
- `results/` - Generated outputs and evaluation results

### Testing the Model

After training, run the testing and evaluation:

```sh
# Generate samples
python generate_text.py --model_path checkpoints/best_model.pth

# Evaluate the results
python evaluate_model.py --samples_path results/generated_samples.json
```
