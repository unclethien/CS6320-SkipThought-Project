import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from transformers import AutoTokenizer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# ----------- Dataset -----------

class WikiTextSentenceDataset(Dataset):
    def __init__(self, tokenizer, split="train", max_length=20):
        self.data = load_dataset("wikitext", "wikitext-103-raw-v1", split=split)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pairs = self._extract_sentence_pairs()

    def _extract_sentence_pairs(self):
        import re
        sents = []
        for line in self.data["text"]:
            line = line.strip()
            if len(line) < 5 or line.startswith("="): continue
            line = re.sub(r"([.!?])", r"\1\n", line)
            sents += [s.strip() for s in line.splitlines() if s.strip()]
        return [(sents[i], sents[i + 1]) for i in range(len(sents) - 1)]

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        curr, nxt = self.pairs[idx]
        curr_ids = self.tokenizer(curr, truncation=True, padding="max_length",
                                  max_length=self.max_length, return_tensors="pt")["input_ids"].squeeze()
        nxt_ids = self.tokenizer(nxt, truncation=True, padding="max_length",
                                 max_length=self.max_length, return_tensors="pt")["input_ids"].squeeze()
        return curr_ids, nxt_ids

class HFVocabWrapper:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.idx2word = [tokenizer.decode([i]) for i in range(tokenizer.vocab_size)]
        self.pad_token_id = tokenizer.pad_token_id
        self.eos_token_id = tokenizer.eos_token_id

    def __getitem__(self, item):
        if item == "<pad>":
            return self.pad_token_id
        if item == "<eos>":
            return self.eos_token_id
        if isinstance(item, str):
            return self.tokenizer.encode(item)[0]
        return self.idx2word[item]

    def __len__(self):
        return self.tokenizer.vocab_size

# ----------- Model Components -----------

class SkipThoughtEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim=300, hidden_dim=2400, bidirectional=True):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.gru = nn.GRU(embed_dim, hidden_dim // 2 if bidirectional else hidden_dim,
                          batch_first=True, bidirectional=bidirectional)

    def forward(self, x):
        embedded = self.embedding(x)
        _, h = self.gru(embedded)
        if self.gru.bidirectional:
            h = torch.cat([h[-2], h[-1]], dim=1)
        else:
            h = h[-1]
        return h

class Generator(nn.Module):
    def __init__(self, vocab_size, embed_dim=300, hidden_dim=2400):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, vocab_size)

    def forward(self, tgt, sentence_vec):
        embedded = self.embedding(tgt)
        h0 = sentence_vec.unsqueeze(0)
        out, _ = self.gru(embedded, h0)
        return self.fc_out(out)

class Discriminator(nn.Module):
    def __init__(self, vocab_size, embed_dim=300, hidden_dim=1024):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        embedded = self.embedding(x)
        _, h = self.gru(embedded)
        return self.fc(h[-1])

# ----------- Training Loop -----------

def train_wgan_skipthought(
    encoder, generator, discriminator,
    dataloader, vocab,
    g_optimizer, d_optimizer,
    num_epochs=5, n_critic=5, clip_value=0.01, device="cuda"):

    encoder.to(device)
    generator.to(device)
    discriminator.to(device)

    loss_fn = nn.CrossEntropyLoss(ignore_index=vocab["<pad>"])

    for epoch in range(num_epochs):
        total_d_loss, total_g_loss, total_bleu = 0.0, 0.0, 0.0

        for i, (curr_sent, next_sent) in enumerate(dataloader):
            curr_sent, next_sent = curr_sent.to(device), next_sent.to(device)

            with torch.no_grad():
                sent_vec = encoder(curr_sent)

            # --- Train Discriminator ---
            for _ in range(n_critic):
                d_optimizer.zero_grad()
                real_scores = discriminator(next_sent)

                with torch.no_grad():
                    fake_logits = generator(next_sent[:, :-1], sent_vec)
                    fake_tokens = torch.argmax(fake_logits, dim=-1)

                fake_scores = discriminator(fake_tokens)
                d_loss = -torch.mean(real_scores) + torch.mean(fake_scores)
                d_loss.backward()
                d_optimizer.step()

                for p in discriminator.parameters():
                    p.data.clamp_(-clip_value, clip_value)

            # --- Train Generator ---
            g_optimizer.zero_grad()
            fake_logits = generator(next_sent[:, :-1], sent_vec)
            fake_tokens = torch.argmax(fake_logits, dim=-1)
            g_score = discriminator(fake_tokens)

            g_loss = -torch.mean(g_score)
            token_loss = loss_fn(fake_logits.reshape(-1, fake_logits.size(-1)), next_sent[:, 1:].reshape(-1))
            total_loss = g_loss + 0.1 * token_loss
            total_loss.backward()
            g_optimizer.step()

            # --- BLEU Eval ---
            smooth_fn = SmoothingFunction().method1
            for ref, gen in zip(next_sent[:, 1:], fake_tokens):
                ref_tokens = [vocab[idx.item()] for idx in ref if idx.item() not in [vocab["<pad>"], vocab["<eos>"]]]
                gen_tokens = [vocab[idx.item()] for idx in gen if idx.item() not in [vocab["<pad>"], vocab["<eos>"]]]
                total_bleu += sentence_bleu([ref_tokens], gen_tokens, smoothing_function=smooth_fn)

            total_d_loss += d_loss.item()
            total_g_loss += g_loss.item()

        print(f"Epoch {epoch+1}/{num_epochs} | D_loss: {total_d_loss:.4f} | G_loss: {total_g_loss:.4f} | BLEU: {total_bleu/len(dataloader.dataset):.4f}")

# ----------- Run -----------

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token  # <- Fix: Add pad token

    vocab = HFVocabWrapper(tokenizer)
    dataset = WikiTextSentenceDataset(tokenizer, split="train[:30%]")  # quick testing
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    encoder = SkipThoughtEncoder(len(vocab)).to(device)
    generator = Generator(len(vocab)).to(device)
    discriminator = Discriminator(len(vocab)).to(device)

    g_optimizer = optim.Adam(generator.parameters(), lr=1e-4)
    d_optimizer = optim.Adam(discriminator.parameters(), lr=1e-4)

    train_wgan_skipthought(encoder, generator, discriminator, dataloader, vocab, g_optimizer, d_optimizer, num_epochs=5, device=device)

#----result from run -----
###
#$ python trying.py
#Epoch 1/5 | D_loss: -1610957.6603 | G_loss: 825384.8040 | BLEU: 0.0334
#Epoch 2/5 | D_loss: -1599969.5199 | G_loss: 826346.3715 | BLEU: 0.0473
#Epoch 3/5 | D_loss: -1597037.6307 | G_loss: 825441.1195 | BLEU: 0.0553
#Epoch 4/5 | D_loss: -1596917.9053 | G_loss: 825636.2611 | BLEU: 0.0619
#Epoch 5/5 | D_loss: -1596878.5875 | G_loss: 824285.8029 | BLEU: 0.0678
