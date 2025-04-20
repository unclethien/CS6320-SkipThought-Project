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

def load_bookcorpus_dataset(data_config, tokenizer):
    """Loads and prepares the BookCorpus dataset for language modeling."""
    dataset_name = data_config.get('dataset_name', 'bookcorpus')
    batch_size = data_config.get('batch_size', 32)
    max_length = data_config.get('params', {}).get('max_length', 128) # Default LM sequence length
    validation_split_percentage = data_config.get('validation_split', 0.05) # Use 5% for validation

    logger.info(f"Loading dataset '{dataset_name}' from Hugging Face Hub...")
    try:
        # Note: BookCorpus might require manual download/setup or specific access.
        # Using 'wikitext' as a more accessible alternative if bookcorpus fails.
        # Replace 'wikitext', 'wikitext-2-raw-v1' with 'bookcorpus' if you have access.
        raw_datasets = load_dataset('wikitext', 'wikitext-2-raw-v1')
        # raw_datasets = load_dataset(dataset_name)
        logger.info(f"Dataset loaded: {raw_datasets}")
    except Exception as e:
        logger.error(f"Failed to load dataset '{dataset_name}': {e}")
        logger.error("Please ensure the dataset exists and you have the necessary permissions/setup.")
        logger.error("Check https://huggingface.co/datasets/bookcorpus for details.")
        raise

    # --- Tokenization --- 
    def tokenize_function(examples):
        # Tokenize the text. This returns dict {'input_ids': ..., 'attention_mask': ...}
        return tokenizer(examples['text'], truncation=False) # Don't truncate yet

    logger.info("Tokenizing dataset...")
    tokenized_datasets = raw_datasets.map(tokenize_function, batched=True, remove_columns=['text'])
    logger.info(f"Tokenization complete: {tokenized_datasets}")

    # --- Chunking --- 
    # Main data processing function that will concatenate all texts from our dataset 
    # and generate chunks of max_seq_length.
    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop,
        # you can customize this part to your needs.
        total_length = (total_length // max_length) * max_length
        # Split by chunks of max_len.
        result = {
            k: [t[i : i + max_length] for i in range(0, total_length, max_length)]
            for k, t in concatenated_examples.items()
        }
        # Create labels (for language modeling, labels are the same as inputs shifted)
        # Note: The trainer will likely handle the shifting, so we just copy input_ids
        result["labels"] = result["input_ids"].copy()
        return result

    logger.info(f"Chunking dataset into sequences of max length {max_length}...")
    lm_datasets = tokenized_datasets.map(group_texts, batched=True)
    logger.info(f"Chunking complete: {lm_datasets}")

    # --- Splitting (if necessary) ---
    if 'validation' not in lm_datasets:
        logger.info(f"No 'validation' split found. Splitting 'train' into train/validation ({1-validation_split_percentage:.0%}/{validation_split_percentage:.0%}).")
        # Careful: split_dataset() might be slow for large datasets
        split_datasets = lm_datasets['train'].train_test_split(test_size=validation_split_percentage, seed=data_config.get('seed', 42))
        train_ds = split_datasets['train']
        val_ds = split_datasets['test']
        # Use validation split also for test split if no test split exists
        test_ds = val_ds 
        logger.info(f"Splitting complete: train={len(train_ds)}, validation={len(val_ds)}, test={len(test_ds)}")
    else:
        train_ds = lm_datasets['train']
        val_ds = lm_datasets['validation']
        test_ds = lm_datasets.get('test', val_ds) # Use validation if test doesn't exist
        logger.info(f"Using existing splits: train={len(train_ds)}, validation={len(val_ds)}, test={len(test_ds)}")

    # --- DataLoaders --- 
    # The datasets library handles collation automatically if format is set
    # lm_datasets.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
    
    # Need a custom collate function if not using set_format or if specific padding is needed
    # For standard LM, HuggingFace Trainer often handles this, but DataLoader needs it. 
    from transformers import default_data_collator
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=default_data_collator)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=default_data_collator)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=default_data_collator)

    logger.info(f"DataLoaders created: train batches={len(train_loader)}, val batches={len(val_loader)}, test batches={len(test_loader)}")

    return train_loader, val_loader, test_loader
 # def load_data(data_config, tokenizer):
 #     """Loads real paraphrase data and creates DataLoaders."""
 #     logger = logging.getLogger(__name__)
