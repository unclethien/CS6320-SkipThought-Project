"""
Adapter module to import components from Second-Skip-Though-GAN.py
This resolves the Python module naming issue with hyphens
"""
import os
import sys
import importlib.util

# Get the absolute path to the module file
module_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Second-Skip-Though-GAN.py")

# Load the module from file path
spec = importlib.util.spec_from_file_location("skip_thought_gan_module", module_path)
skip_thought_gan = importlib.util.module_from_spec(spec)
sys.modules["skip_thought_gan_module"] = skip_thought_gan
spec.loader.exec_module(skip_thought_gan)

# Export the classes and functions we need
Generator = skip_thought_gan.Generator
QNetwork = skip_thought_gan.QNetwork
Discriminator = skip_thought_gan.Discriminator
compute_diversity_metrics = skip_thought_gan.compute_diversity_metrics
calculate_sentence_bleu = skip_thought_gan.calculate_sentence_bleu
generate_text_from_latent = skip_thought_gan.generate_text_from_latent

# Import the enhanced generator that's compatible with the saved checkpoint
from enhanced_generator import EnhancedGenerator