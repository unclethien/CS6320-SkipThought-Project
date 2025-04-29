# Install required packages first
# !pip install torch transformers datasets nltk bert-score accelerate

import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer, BertConfig
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
import nltk
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from bert_score import score as bert_score
import numpy as np

# Memory optimization setup
nltk.download('punkt')
torch.backends.cudnn.benchmark = True
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ================== Configuration ==================
class Config:
    batch_size = 4
    max_length = 32
    grad_accum_steps = 2
    eval_samples = 50
    bert_model = 'bert-base-uncased'
    fp16 = True
    noise_dim = 64

# ================== Model Components ==================
class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained(Config.bert_model)
        self.bert.gradient_checkpointing_enable()
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state, outputs.pooler_output

class LiteBertDecoder(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        config = BertConfig.from_pretrained(Config.bert_model)
        config.num_hidden_layers = 3
        config.is_decoder = True
        config.add_cross_attention = True
        self.bert = BertModel(config)
        self.fc = nn.Linear(config.hidden_size, vocab_size)
        
    def forward(self, input_ids, encoder_hidden_states, attention_mask=None, encoder_attention_mask=None):
        outputs = self.bert(
            input_ids=input_ids,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=attention_mask,
            encoder_attention_mask=encoder_attention_mask
        )
        return self.fc(outputs.last_hidden_state)

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(Config.noise_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 768)
        )
    
    def forward(self, z):
        return self.net(z)

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(768, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 1)
        )
    
    def forward(self, x):
        return self.net(x)

# ================== Full Model Wrapper ==================
class GANBert(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder().to(device)
        self.decoder = LiteBertDecoder(tokenizer.vocab_size).to(device)
        self.generator = Generator().to(device)
        self.discriminator = Discriminator().to(device)
        
        self.optim_ae = torch.optim.Adam(
            list(self.encoder.parameters()) + list(self.decoder.parameters()),
            lr=2e-5
        )
        self.optim_gen = torch.optim.Adam(self.generator.parameters(), lr=1e-4)
        self.optim_dis = torch.optim.Adam(self.discriminator.parameters(), lr=1e-4)

# ================== Dataset & Tokenizer ==================
tokenizer = BertTokenizer.from_pretrained(Config.bert_model)

class SentencePairDataset(Dataset):
    def __init__(self, texts):
        self.sentence_pairs = []
        for text in texts:
            sentences = [s.strip() for s in nltk.sent_tokenize(text) if len(s.strip()) > 0]
            for i in range(len(sentences)-1):
                self.sentence_pairs.append((sentences[i], sentences[i+1]))
    
    def __len__(self):
        return len(self.sentence_pairs)
    
    def __getitem__(self, idx):
        input_sent, target_sent = self.sentence_pairs[idx]
        enc = tokenizer(
            input_sent, 
            padding='max_length',
            truncation=True,
            max_length=Config.max_length,
            return_tensors='pt'
        )
        dec = tokenizer(
            target_sent,
            padding='max_length',
            truncation=True,
            max_length=Config.max_length,
            return_tensors='pt'
        )
        return {k: v.squeeze(0) for k, v in enc.items()}, {k: v.squeeze(0) for k, v in dec.items()}
    
# ================== Evaluation Functions ==================
def evaluate_model(model, loader, num_samples=50):
    references = []
    hypotheses = []
    smooth = SmoothingFunction()
    
    model.eval()
    with torch.no_grad():
        for idx, (enc, dec) in enumerate(loader):
            if idx >= num_samples: break
            
            batch_size = enc['input_ids'].size(0)
            enc_input_ids = enc['input_ids'].to(device)
            enc_attention_mask = enc['attention_mask'].to(device)
            
            # Get encoder outputs
            encoder_hidden, _ = model.encoder(enc_input_ids, enc_attention_mask)
            
            # Initialize predictions
            pred_ids = torch.full((batch_size, 1), tokenizer.cls_token_id, 
                                dtype=torch.long, device=device)
            
            for step in range(Config.max_length - 1):
                # Create proper masks
                decoder_attention_mask = (pred_ids != tokenizer.pad_token_id).long().to(device)
                
                outputs = model.decoder(
                    input_ids=pred_ids,
                    encoder_hidden_states=encoder_hidden,
                    attention_mask=decoder_attention_mask,
                    encoder_attention_mask=enc_attention_mask  # Use original 2D mask
                )
                
                next_tokens = outputs.argmax(-1)[:, -1].unsqueeze(1)
                pred_ids = torch.cat([pred_ids, next_tokens], dim=1)
                
                if (next_tokens == tokenizer.sep_token_id).all():
                    break
            
            # Get actual references from dataset
            true_texts = [tokenizer.decode(ids, skip_special_tokens=True) 
                         for ids in dec['input_ids'].cpu().numpy()]
            
            # Get predictions
            pred_texts = [tokenizer.decode(ids, skip_special_tokens=True) 
                         for ids in pred_ids.cpu().numpy()]

            # Format for metrics
            references.extend([[text.split()] for text in true_texts])
            hypotheses.extend([text.split() for text in pred_texts])
    
    # Calculate BLEU scores
    bleu_scores = {
        'BLEU-1': corpus_bleu(references, hypotheses, weights=(1,0,0,0), smoothing_function=smooth.method4),
        'BLEU-2': corpus_bleu(references, hypotheses, weights=(0.5,0.5,0,0), smoothing_function=smooth.method4),
        'BLEU-3': corpus_bleu(references, hypotheses, weights=(0.33,0.33,0.33,0), smoothing_function=smooth.method4),
        'BLEU-4': corpus_bleu(references, hypotheses, weights=(0.25,0.25,0.25,0.25), smoothing_function=smooth.method4)
    }
    
    # Calculate BERTScore
    P, R, F1 = bert_score([' '.join(h) for h in hypotheses],
                         [' '.join(r[0]) for r in references],
                         lang='en')
    
    return bleu_scores, np.mean(F1.numpy())

# ================== Training Loop ==================
def train(model, train_loader):
    scaler = torch.cuda.amp.GradScaler(enabled=Config.fp16)
    criterion_ae = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    criterion_gan = nn.BCEWithLogitsLoss()
    
    for epoch in range(10):
        model.train()
        for batch_idx, (enc, dec) in enumerate(train_loader):
            with torch.cuda.amp.autocast(enabled=Config.fp16):
                # Forward pass
                enc_input_ids = enc['input_ids'].squeeze().to(device)
                enc_attention_mask = enc['attention_mask'].squeeze().to(device)
                dec_input_ids = dec['input_ids'].squeeze().to(device)
                
                # Autoencoder
                encoder_hidden, encoder_pooled = model.encoder(enc_input_ids, enc_attention_mask)
                outputs = model.decoder(dec_input_ids[:, :-1], encoder_hidden)
                loss_ae = criterion_ae(outputs.view(-1, tokenizer.vocab_size),
                                      dec_input_ids[:, 1:].contiguous().view(-1))
                
                # GAN training
                real_labels = torch.ones(enc_input_ids.size(0), 1).to(device)
                fake_labels = torch.zeros(enc_input_ids.size(0), 1).to(device)
                
                # Discriminator
                real_emb = encoder_pooled.detach()
                z = torch.randn(enc_input_ids.size(0), Config.noise_dim).to(device)
                fake_emb = model.generator(z).detach()
                
                real_loss = criterion_gan(model.discriminator(real_emb), real_labels)
                fake_loss = criterion_gan(model.discriminator(fake_emb), fake_labels)
                loss_dis = (real_loss + fake_loss) / 2
                
                # Generator
                z = torch.randn(enc_input_ids.size(0), Config.noise_dim).to(device)
                fake_emb = model.generator(z)
                loss_gen = criterion_gan(model.discriminator(fake_emb), real_labels)
                
                total_loss = (loss_ae + loss_dis + loss_gen) / Config.grad_accum_steps
                
            scaler.scale(total_loss).backward()
            
            if (batch_idx + 1) % Config.grad_accum_steps == 0:
                scaler.step(model.optim_ae)
                scaler.step(model.optim_gen)
                scaler.step(model.optim_dis)
                scaler.update()
                model.optim_ae.zero_grad()
                model.optim_gen.zero_grad()
                model.optim_dis.zero_grad()
        
        # Evaluation
        bleu_scores, bert_f1 = evaluate_model(model, train_loader)
        print(f"\nEpoch {epoch+1}:")
        print(f"BLEU Scores: {bleu_scores}")
        print(f"BERTScore F1: {bert_f1:.3f}")

# ================== Generation Function ==================
def generate_examples(model, num_examples=3):
    dataset = load_dataset('bookcorpus', split='train[:70%]')['text']
    inputs = [nltk.sent_tokenize(text)[0] for text in dataset[:num_examples]]
    
    model.eval()
    with torch.no_grad():
        for input_text in inputs:
            encoding = tokenizer(input_text, return_tensors='pt',
                               padding='max_length', truncation=True,
                               max_length=Config.max_length).to(device)
            encoder_hidden, _ = model.encoder(encoding['input_ids'], encoding['attention_mask'])
            
            pred_ids = torch.full((1, 1), tokenizer.cls_token_id, 
                                dtype=torch.long, device=device)
            
            for step in range(Config.max_length - 1):
                decoder_mask = (pred_ids != tokenizer.pad_token_id).long().to(device)
                
                outputs = model.decoder(
                    pred_ids,
                    encoder_hidden,
                    attention_mask=decoder_mask,
                    encoder_attention_mask=encoding['attention_mask']  # 2D mask
                )
                next_token = outputs.argmax(-1)[:, -1].unsqueeze(1)
                pred_ids = torch.cat([pred_ids, next_token], dim=1)
                if next_token == tokenizer.sep_token_id:
                    break
            
            print(f"\nInput: {input_text}")
            print(f"Generated: {tokenizer.decode(pred_ids[0], skip_special_tokens=True)}")
# ================== Main Execution ==================
if __name__ == "__main__":
    # Initialize components
    train_data = load_dataset('bookcorpus', split='train[:50%]')['text']
    train_dataset = SentencePairDataset(train_data)
    train_loader = DataLoader(train_dataset, batch_size=Config.batch_size, shuffle=True)
    
    model = GANBert()
    
    # Train
    train(model, train_loader)
    
    # Generate examples
    generate_examples(model)