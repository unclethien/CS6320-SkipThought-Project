import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer
import logging

logger = logging.getLogger(__name__)

class SentenceEncoder(nn.Module):
    """Wrapper for using pre-trained Sentence Transformers as the Encoder.

    Loads a specified Sentence Transformer model and provides an interface
    to encode batches of text into fixed-size embeddings (latent vectors).
    """
    def __init__(self, model_name: str, device: str = 'cpu', freeze: bool = True):
        """
        Args:
            model_name: Name of the Sentence Transformer model to load 
                        (e.g., 'all-MiniLM-L6-v2').
            device: The device ('cpu' or 'cuda') to run the model on.
            freeze: If True, freeze the weights of the pre-trained model.
        """
        super().__init__()
        self.device = device
        logger.info(f"Loading Sentence Transformer model: {model_name}...")
        try:
            self.st_model = SentenceTransformer(model_name, device=device)
        except Exception as e:
            logger.error(f"Failed to load Sentence Transformer model '{model_name}'. Error: {e}")
            logger.error("Please ensure the model name is correct and dependencies are installed ('pip install sentence-transformers').")
            raise
        
        self.latent_dim = self.st_model.get_sentence_embedding_dimension()
        logger.info(f"Sentence Transformer model loaded. Latent dimension: {self.latent_dim}")

        if freeze:
            logger.info("Freezing Sentence Transformer model weights.")
            for param in self.st_model.parameters():
                param.requires_grad = False
        else:
            logger.info("Sentence Transformer model weights are trainable (unfrozen).")

    def forward(self, sentences: list[str]) -> torch.Tensor:
        """
        Encodes a batch of sentences into latent vectors.

        Args:
            sentences: A list of sentences (strings).

        Returns:
            A tensor containing the sentence embeddings.
            Shape: [batch_size, latent_dim].
        """
        # The encode method handles batching internally and returns numpy arrays by default.
        # Convert to tensor and move to the specified device.
        # Set convert_to_tensor=True for direct tensor output.
        embeddings = self.st_model.encode(
            sentences, 
            convert_to_tensor=True, 
            device=self.device, # Ensure encoding happens on the correct device
            show_progress_bar=False # Disable internal progress bar for cleaner logs
        )
        return embeddings

    def to(self, device):
        # Ensure the internal model is moved to the correct device when .to() is called
        super().to(device)
        self.st_model.to(device)
        self.device = device
        return self
    
# Example usage (for testing)
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    # --- Configuration --- 
    # Use a small, fast model for quick testing if needed, 
    # but default to the one likely in config
    # model_to_test = 'paraphrase-MiniLM-L3-v2' # Smaller/faster alternative
    model_to_test = 'all-MiniLM-L6-v2' 
    test_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    freeze_weights = True
    # ---------------------

    logger.info(f"Testing SentenceEncoder with model: {model_to_test} on device: {test_device}")
    
    try:
        encoder = SentenceEncoder(model_name=model_to_test, device=test_device, freeze=freeze_weights)
        encoder.to(test_device) # Ensure model is on the right device

        print("\nEncoder Architecture (Underlying Sentence Transformer):")
        # Print the first few modules of the underlying model for info
        print(encoder.st_model)

        logger.info(f"Encoder Latent Dimension: {encoder.latent_dim}")

        test_sentences = [
            "This is the first sentence.",
            "Here is another example sentence.",
            "Sentence transformers are useful for getting embeddings.",
            "A short one."
        ]
        
        # Check if model is frozen status
        is_frozen = all(not p.requires_grad for p in encoder.st_model.parameters())
        logger.info(f"Encoder weights frozen: {is_frozen} (Expected: {freeze_weights})")
        assert is_frozen == freeze_weights

        # Encode the sentences
        with torch.no_grad() if freeze_weights else torch.enable_grad():
             embeddings = encoder(test_sentences)

        logger.info(f"Input Sentences (Batch Size): {len(test_sentences)}")
        logger.info(f"Output Embeddings Shape: {embeddings.shape}")
        logger.info(f"Output Embeddings Device: {embeddings.device}")

        assert embeddings.shape == (len(test_sentences), encoder.latent_dim)
        assert str(embeddings.device).startswith(test_device)

        logger.info("\nSentenceEncoder test passed!")

    except Exception as e:
        logger.error(f"Error during SentenceEncoder test: {e}", exc_info=True)
