import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from transformers import AutoTokenizer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt')


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ---------------------- Model Definitions ----------------------

class SkipThoughtEncoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.gru = nn.GRU(embedding_dim, hidden_dim, batch_first=True)

    def forward(self, x):
        emb = self.embedding(x)
        _, h = self.gru(emb)
        return h.squeeze(0)


class SkipThoughtDecoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.gru = nn.GRU(embedding_dim + hidden_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, input_seq, context):
        embedded = self.embedding(input_seq)
        context = context.unsqueeze(1).expand(-1, embedded.size(1), -1)
        rnn_input = torch.cat([embedded, context], dim=2)
        output, _ = self.gru(rnn_input)
        return self.fc(output)



class Generator(nn.Module):
    def __init__(self, noise_dim, hidden_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(noise_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, z):
        return self.model(z)


class Discriminator(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, vec):
        return self.model(vec)


# ---------------------- Dataset ----------------------

class BookCorpusTripletDataset(Dataset):
    def __init__(self, texts, tokenizer, seq_len=64):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.examples = []
        for i in range(1, len(texts) - 1):
            prev = tokenizer(texts[i - 1], padding='max_length', truncation=True, max_length=seq_len, return_tensors="pt")
            curr = tokenizer(texts[i], padding='max_length', truncation=True, max_length=seq_len, return_tensors="pt")
            nxt = tokenizer(texts[i + 1], padding='max_length', truncation=True, max_length=seq_len, return_tensors="pt")
            self.examples.append((prev['input_ids'].squeeze(), curr['input_ids'].squeeze(), nxt['input_ids'].squeeze()))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


# ---------------------- Training Function ----------------------

def train_epoch(loader, encoder, decoder_prev, decoder_next, G, D,
                enc_opt, dec_opt, G_opt, D_opt, vocab_size, noise_dim, device):

    encoder.train()
    decoder_prev.train()
    decoder_next.train()
    G.train()
    D.train()

    for prev, curr, nxt in loader:
        prev, curr, nxt = prev.to(device), curr.to(device), nxt.to(device)

        real_vec = encoder(curr)

        pred_prev = decoder_prev(prev[:, :-1], real_vec)
        pred_next = decoder_next(nxt[:, :-1], real_vec)

        loss_prev = F.cross_entropy(pred_prev.view(-1, vocab_size), prev[:, 1:].reshape(-1), ignore_index=0)
        loss_next = F.cross_entropy(pred_next.view(-1, vocab_size), nxt[:, 1:].reshape(-1), ignore_index=0)

        enc_loss = loss_prev + loss_next
        enc_opt.zero_grad()
        dec_opt.zero_grad()
        enc_loss.backward()
        enc_opt.step()
        dec_opt.step()

        z = torch.randn(real_vec.size(0), noise_dim).to(device)
        fake_vec = G(z)
        G_loss = F.binary_cross_entropy_with_logits(D(fake_vec), torch.ones_like(D(fake_vec)))

        G_opt.zero_grad()
        G_loss.backward()
        G_opt.step()

        real_labels = torch.ones_like(D(real_vec))
        fake_labels = torch.zeros_like(D(fake_vec.detach()))

        D_real_loss = F.binary_cross_entropy_with_logits(D(real_vec.detach()), real_labels)
        D_fake_loss = F.binary_cross_entropy_with_logits(D(fake_vec.detach()), fake_labels)
        D_loss = D_real_loss + D_fake_loss

        D_opt.zero_grad()
        D_loss.backward()
        D_opt.step()

        print(f"Enc Loss: {enc_loss.item():.3f} | G Loss: {G_loss.item():.3f} | D Loss: {D_loss.item():.3f}")


# ---------------------- Evaluation Function ----------------------

def decode_ids(tokenizer, ids):
    return tokenizer.decode(ids, skip_special_tokens=True)

def evaluate_bleu(test_loader, encoder, decoder_prev, decoder_next, tokenizer, device):
    encoder.eval()
    decoder_prev.eval()
    decoder_next.eval()

    smoothie = SmoothingFunction().method1

    scores_1, scores_2, scores_3, scores_4 = [], [], [], []

    with torch.no_grad():
        for prev_ids, curr_ids, next_ids in test_loader:
            prev_ids, curr_ids, next_ids = prev_ids.to(device), curr_ids.to(device), next_ids.to(device)
            context = encoder(curr_ids)

            prev_logits = decoder_prev(prev_ids[:, :-1], context)
            next_logits = decoder_next(next_ids[:, :-1], context)

            prev_preds = prev_logits.argmax(dim=-1)
            next_preds = next_logits.argmax(dim=-1)

            for ref_ids, hyp_ids in zip(prev_ids[:, 1:], prev_preds):
                ref = word_tokenize(decode_ids(tokenizer, ref_ids))
                hyp = word_tokenize(decode_ids(tokenizer, hyp_ids))
                scores_1.append(sentence_bleu([ref], hyp, weights=(1, 0, 0, 0), smoothing_function=smoothie))
                scores_2.append(sentence_bleu([ref], hyp, weights=(0.5, 0.5, 0, 0), smoothing_function=smoothie))
                scores_3.append(sentence_bleu([ref], hyp, weights=(0.33, 0.33, 0.33, 0), smoothing_function=smoothie))
                scores_4.append(sentence_bleu([ref], hyp, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothie))

            for ref_ids, hyp_ids in zip(next_ids[:, 1:], next_preds):
                ref = word_tokenize(decode_ids(tokenizer, ref_ids))
                hyp = word_tokenize(decode_ids(tokenizer, hyp_ids))
                scores_1.append(sentence_bleu([ref], hyp, weights=(1, 0, 0, 0), smoothing_function=smoothie))
                scores_2.append(sentence_bleu([ref], hyp, weights=(0.5, 0.5, 0, 0), smoothing_function=smoothie))
                scores_3.append(sentence_bleu([ref], hyp, weights=(0.33, 0.33, 0.33, 0), smoothing_function=smoothie))
                scores_4.append(sentence_bleu([ref], hyp, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothie))

    print("\n--- BLEU Scores ---")
    print(f"BLEU-1: {sum(scores_1)/len(scores_1):.4f}")
    print(f"BLEU-2: {sum(scores_2)/len(scores_2):.4f}")
    print(f"BLEU-3: {sum(scores_3)/len(scores_3):.4f}")
    print(f"BLEU-4: {sum(scores_4)/len(scores_4):.4f}")


# ---------------------- Main Script ----------------------

def main():
    embedding_dim = 620
    hidden_dim = 1600
    noise_dim = 100
    seq_len = 64
    batch_size = 16
    num_epochs = 5

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    vocab_size = tokenizer.vocab_size

    dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")
    texts = [t for t in dataset["text"] if len(t.strip().split()) > 5]
    train_dataset = BookCorpusTripletDataset(texts[:50000], tokenizer, seq_len=seq_len)
    test_dataset = BookCorpusTripletDataset(texts[50000:52000], tokenizer, seq_len=seq_len)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=8)

    encoder = SkipThoughtEncoder(vocab_size, embedding_dim, hidden_dim).to(device)
    decoder_prev = SkipThoughtDecoder(vocab_size, embedding_dim, hidden_dim).to(device)
    decoder_next = SkipThoughtDecoder(vocab_size, embedding_dim, hidden_dim).to(device)
    G = Generator(noise_dim, hidden_dim).to(device)
    D = Discriminator(hidden_dim).to(device)

    enc_opt = optim.Adam(encoder.parameters(), lr=1e-4)
    dec_opt = optim.Adam(list(decoder_prev.parameters()) + list(decoder_next.parameters()), lr=1e-4)
    G_opt = optim.Adam(G.parameters(), lr=1e-4)
    D_opt = optim.Adam(D.parameters(), lr=1e-4)

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        train_epoch(train_loader, encoder, decoder_prev, decoder_next, G, D,
                    enc_opt, dec_opt, G_opt, D_opt, vocab_size, noise_dim, device)

    evaluate_bleu(test_loader, encoder, decoder_prev, decoder_next, tokenizer, device)


if __name__ == '__main__':
    main()
