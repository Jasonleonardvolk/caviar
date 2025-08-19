from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
# {PROJECT_ROOT}\models\efficientnet\train_saigon.py

import torch
import torch.nn as nn
import torch.optim as optim
import json
import os
import random

CORPUS_PATH = "corpus.txt"
VOCAB_PATH = "vocab.json"
MODEL_PATH = "saigon_lstm.pt"
HIDDEN_SIZE = 256
NUM_LAYERS = 3
EPOCHS = 15  # More epochs = better for small data!
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32
SEQ_LEN = 160

class CharLSTM(nn.Module):
    def __init__(self, vocab_size, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden=None):
        x = self.embed(x)
        out, hidden = self.lstm(x, hidden)
        logits = self.linear(out)
        return logits, hidden

# -- 1. Load Corpus
with open(CORPUS_PATH, encoding="utf-8") as f:
    lines = [line.strip() for line in f if line.strip()]
text = "\n".join(lines)

# -- 2. Build vocab
chars = sorted(list(set(text)))
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {str(i): ch for i, ch in enumerate(chars)}
vocab = {"stoi": stoi, "itos": itos}
with open(VOCAB_PATH, "w", encoding="utf-8") as f:
    json.dump(vocab, f)

# -- 3. Encode text
idxs = [stoi[c] for c in text]
data = torch.tensor(idxs, dtype=torch.long)
val_split = 0.05
val_len = int(len(data) * val_split)
train_data, val_data = data[:-val_len], data[-val_len:]

def get_batch(split="train"):
    data_split = train_data if split=="train" else val_data
    starts = torch.randint(0, len(data_split) - SEQ_LEN - 1, (BATCH_SIZE,))
    x = torch.stack([data_split[s:s+SEQ_LEN] for s in starts]).to(DEVICE)
    y = torch.stack([data_split[s+1:s+SEQ_LEN+1] for s in starts]).to(DEVICE)
    return x, y

# -- 4. Setup model
model = CharLSTM(len(chars)).to(DEVICE)
optimizer = optim.AdamW(model.parameters(), lr=1.2e-3, weight_decay=1e-2)
criterion = nn.CrossEntropyLoss()
best_val_loss = float("inf")
epochs_since_improve = 0

print(f"Training CharLSTM: {len(text)} chars, {len(chars)} vocab, device: {DEVICE}")

# -- 5. Training loop (with validation, early stopping)
for epoch in range(EPOCHS):
    model.train()
    losses = []
    for step in range(0, max(1, len(train_data) // (BATCH_SIZE * SEQ_LEN))):
        x, y = get_batch("train")
        logits, _ = model(x)
        loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 2.0)  # Prevent explosions
        optimizer.step()
        losses.append(loss.item())
    avg_loss = sum(losses)/len(losses)

    # Validation
    model.eval()
    val_losses = []
    with torch.no_grad():
        for _ in range(5):
            x, y = get_batch("val")
            logits, _ = model(x)
            val_loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
            val_losses.append(val_loss.item())
    avg_val_loss = sum(val_losses)/len(val_losses)

    print(f"Epoch {epoch+1}: train={avg_loss:.4f}  val={avg_val_loss:.4f}")

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), MODEL_PATH)
        print("Best model saved.")
        epochs_since_improve = 0
    else:
        epochs_since_improve += 1

    if epochs_since_improve > 3:
        print("Early stopping: No val improvement.")
        break

print(f"Training done! Best val loss: {best_val_loss:.4f}. Model saved as {MODEL_PATH}")
