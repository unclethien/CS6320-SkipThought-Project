from torch.utils.data import Dataset, DataLoader
import torch
import os
from transformers import AutoTokenizer
from datasets import load_dataset
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def get_tokenizer(tokenizer_name='bert-base-uncased'):
    """Loads a tokenizer."""
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        return tokenizer
    except Exception as e:
        print(f"Error loading tokenizer '{tokenizer_name}': {e}")
        print("Please ensure 'transformers' library is installed and model name is correct.")
        # Fallback to a simple split tokenizer for basic functionality if needed
        class SimpleTokenizer:
            def __init__(self):
                # Placeholder vocab for basic functionality
                self.vocab = {'[PAD]': 0, '[UNK]': 1, '[CLS]': 2, '[SEP]': 3, 'hello': 4, 'world': 5}
                self.ids_to_tokens = {v: k for k, v in self.vocab.items()}
                self.vocab_size = len(self.vocab)
                self.pad_token_id = 0
                self.cls_token_id = 2
                self.sep_token_id = 3
                self.all_special_tokens = ['[PAD]', '[UNK]', '[CLS]', '[SEP]']
            def encode(self, text, add_special_tokens=True, max_length=10, truncation=True, padding='max_length', return_tensors=None):
                tokens = text.lower().split()
                ids = [self.vocab.get(t, self.vocab['[UNK]']) for t in tokens]
                if add_special_tokens:
                    ids = [self.cls_token_id] + ids + [self.sep_token_id]
                if truncation and len(ids) > max_length:
                    ids = ids[:max_length]
                if padding == 'max_length':
                    ids += [self.pad_token_id] * (max_length - len(ids))
                if return_tensors == 'pt':
                    return {'input_ids': torch.tensor([ids]), 'attention_mask': torch.tensor([[1 if i != self.pad_token_id else 0 for i in ids]])}
                return ids
            def decode(self, token_ids, skip_special_tokens=True):
                tokens = [self.ids_to_tokens.get(i, '[UNK]') for i in token_ids]
                if skip_special_tokens:
                    tokens = [t for t in tokens if t not in self.all_special_tokens]
                return " ".join(tokens)
            def convert_ids_to_tokens(self, ids):
                return [self.ids_to_tokens.get(i, '[UNK]') for i in ids]
        print("Using a fallback simple tokenizer.")
        return SimpleTokenizer()

def load_text_dataset_for_encoding(data_config):
    """Loads and prepares a text dataset for sentence/paragraph encoding."""
    dataset_name = data_config.get('dataset_name', 'wikitext') # Default to wikitext if bookcorpus is unavailable
    dataset_config_name = data_config.get('dataset_config_name', 'wikitext-103-raw-v1' if dataset_name == 'wikitext' else None)
    # Example: use 'wikitext-103-raw-v1' for more data than 'wikitext-2-raw-v1'
    # dataset_config_name = 'wikitext-2-raw-v1'

    batch_size = data_config.get('batch_size', 32)
    text_column = data_config.get('text_column', 'text')
    validation_split_percentage = data_config.get('validation_split', 0.05) # Use 5% for validation
    test_split_percentage = data_config.get('test_split', 0.05)             # Use 5% for test
    seed = data_config.get('seed', 42)
    filter_author = data_config.get('filter_author', None)
    # Note: Filtering by author on large datasets like bookcorpus might require specific handling
    # depending on the dataset structure (metadata columns, etc.) - not implemented here.

    logger.info(f"Loading dataset '{dataset_name}' ({dataset_config_name or 'default'}) from Hugging Face Hub...")
    try:
        # Load the raw dataset
        if dataset_name == 'bookcorpus':
            raw_datasets = load_dataset(dataset_name, trust_remote_code=True)
        elif dataset_config_name:
            raw_datasets = load_dataset(dataset_name, dataset_config_name)
        else:
            raw_datasets = load_dataset(dataset_name)
        logger.info(f"Dataset loaded: {raw_datasets}")

        # --- Select Subset for Testing --- 
        subset_percentage = 0.01 # Use 1% for testing
        primary_split_key = None
        if 'train' in raw_datasets:
            primary_split_key = 'train'
        elif len(raw_datasets.keys()) == 1:
            primary_split_key = list(raw_datasets.keys())[0]
        
        if primary_split_key:
            original_size = len(raw_datasets[primary_split_key])
            subset_size = max(1, int(original_size * subset_percentage)) # Ensure at least 1 sample
            logger.info(f"Selecting {subset_percentage*100:.2f}% ({subset_size} samples) of the '{primary_split_key}' split for testing.")
            # Use shuffle(seed=seed) before select for potentially better representation
            raw_datasets[primary_split_key] = raw_datasets[primary_split_key].shuffle(seed=seed).select(range(subset_size))
        else:
             logger.warning(f"Could not determine primary split to select subset from. Using full dataset.")
        logger.info(f"Dataset size after subset selection: {raw_datasets}")
        
        # --- Dataset Splitting Logic ---
        train_exists = 'train' in raw_datasets
        val_exists = 'validation' in raw_datasets
        test_exists = 'test' in raw_datasets

        if train_exists and not val_exists and not test_exists:
            logger.warning("Dataset has only 'train' split. Creating 'validation' and 'test' splits.")
            # Split train into train_val and test
            train_val_split = raw_datasets['train'].train_test_split(
                test_size=test_split_percentage, seed=seed
            )
            raw_datasets['test'] = train_val_split['test']
            train_val_data = train_val_split['train']

            # Split train_val into train and validation
            # Adjust validation percentage relative to the *remaining* data
            adjusted_val_percentage = validation_split_percentage / (1.0 - test_split_percentage)
            final_split = train_val_data.train_test_split(
                test_size=adjusted_val_percentage, seed=seed
            )
            raw_datasets['train'] = final_split['train']
            raw_datasets['validation'] = final_split['test']
            logger.info(f"Created train/validation/test splits: {raw_datasets}")

        elif train_exists and val_exists and not test_exists:
            logger.warning("Dataset has 'train' and 'validation', but no 'test' split. Creating 'test' from 'train'.")
            split_dataset = raw_datasets['train'].train_test_split(
                test_size=test_split_percentage, seed=seed
            )
            raw_datasets['train'] = split_dataset['train']
            raw_datasets['test'] = split_dataset['test']
            logger.info(f"Split 'train' into train/test: {raw_datasets}")
        
        elif train_exists and not val_exists and test_exists:
             logger.warning("Dataset has 'train' and 'test', but no 'validation' split. Creating 'validation' from 'train'.")
             split_dataset = raw_datasets['train'].train_test_split(
                 test_size=validation_split_percentage, seed=seed
             )
             raw_datasets['train'] = split_dataset['train']
             raw_datasets['validation'] = split_dataset['test']
             logger.info(f"Split 'train' into train/validation: {raw_datasets}")
        
        elif not train_exists:
             # Handle cases where the primary split isn't named 'train'
             # This logic might need refinement depending on datasets encountered
             main_split_key = list(raw_datasets.keys())[0]
             logger.error(f"Dataset missing 'train' split. Cannot reliably create train/val/test from '{main_split_key}'. Please check dataset structure.")
             raise ValueError("Dataset missing 'train' split.")
        else:
            logger.info("Using existing 'train', 'validation', and 'test' splits.")

        # --- Preprocessing --- 
        def preprocess_function(examples):
            # Minimal preprocessing: Ensure text is string, maybe basic cleaning
            # Keep only non-empty lines
            examples[text_column] = [text.strip() for text in examples[text_column] if isinstance(text, str) and text.strip()]
            return examples

        logger.info(f"Preprocessing dataset (column: '{text_column}')...")
        processed_datasets = raw_datasets.map(
            preprocess_function,
            batched=True,
            remove_columns=[col for col in raw_datasets['train'].column_names if col != text_column] # Keep only the text column
        )

        # Filter out empty examples potentially created by preprocessing
        processed_datasets = processed_datasets.filter(lambda example: len(example[text_column]) > 0)

        logger.info(f"Preprocessing complete. Final dataset structure: {processed_datasets}")

        # --- Create DataLoaders --- 
        # We need to collate batches of text strings
        def collate_fn(batch):
            # 'batch' is a list of dicts, e.g., [{'text': 'sentence 1'}, {'text': 'sentence 2'}]
            texts = [item[text_column] for item in batch]
            return texts # Return a list of strings

        train_dataloader = DataLoader(
            processed_datasets['train'], 
            batch_size=batch_size, 
            shuffle=True, 
            collate_fn=collate_fn
        )
        
        eval_dataloader = DataLoader(
            processed_datasets['validation'], 
            batch_size=batch_size, 
            shuffle=False, 
            collate_fn=collate_fn
        )

        test_dataloader = None
        if 'test' in processed_datasets:
            test_dataloader = DataLoader(
                processed_datasets['test'],
                batch_size=batch_size,
                shuffle=False,
                collate_fn=collate_fn
            )
            logger.info(f"Test DataLoader created. Test batches: {len(test_dataloader)}")
        else:
             logger.warning("No 'test' split available after processing. Test DataLoader not created.")

        logger.info(f"DataLoaders created. Train batches: {len(train_dataloader)}, Eval batches: {len(eval_dataloader)}")

        # Placeholder for vocabulary size if needed elsewhere (e.g., for Decoder output layer)
        # This should come from the actual tokenizer used for the Decoder
        vocab_size = data_config.get('model', {}).get('decoder', {}).get('vocab_size', 30522) # Get from config
        
        return train_dataloader, eval_dataloader, test_dataloader, vocab_size

    except Exception as e:
        logger.error(f"Failed to load or process dataset '{dataset_name}': {e}")
        logger.error("Check dataset name, configuration, and availability on Hugging Face Hub.")
        logger.error("Consider using 'wikitext' as a fallback.")
        raise

# Rename the main function for clarity
load_and_prepare_dataset = load_text_dataset_for_encoding

# Example usage (for testing)
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    
    # Mock config for testing
    test_config = {
        'dataset_name': 'wikitext',
        'dataset_config_name': 'wikitext-2-raw-v1', # Small version for quick testing
        'batch_size': 4,
        'text_column': 'text',
        'validation_split': 0.1,
        'test_split': 0.1,
        'seed': 42,
        'model': {
            'decoder': {'vocab_size': 30000} # Example vocab size
        }
    }
    
    try:
        train_dl, eval_dl, test_dl, vocab_size = load_and_prepare_dataset(test_config)
        
        logger.info(f"Successfully loaded data. Vocab size (placeholder): {vocab_size}")
        logger.info("Checking first batch of training data...")
        first_batch = next(iter(train_dl))
        logger.info(f"Type of batch: {type(first_batch)}")
        logger.info(f"Length of batch: {len(first_batch)}")
        logger.info(f"Type of first item: {type(first_batch[0])}")
        logger.info(f"First item:\n{first_batch[0]}") # Corrected f-string
        
        logger.info("\nChecking first batch of validation data...")
        first_eval_batch = next(iter(eval_dl))
        logger.info(f"Type of batch: {type(first_eval_batch)}")
        logger.info(f"Length of batch: {len(first_eval_batch)}")
        logger.info(f"First item:\n{first_eval_batch[0]}") # Corrected f-string
        
        if test_dl:
            logger.info("\nChecking first batch of test data...")
            first_test_batch = next(iter(test_dl))
            logger.info(f"Type of batch: {type(first_test_batch)}")
            logger.info(f"Length of batch: {len(first_test_batch)}")
            logger.info(f"First item:\n{first_test_batch[0]}") # Corrected f-string
        
    except Exception as e: # Added missing except block
        logger.error(f"Error during example usage: {e}", exc_info=True)
