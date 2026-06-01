import torch
import torch.nn as nn
from torch.nn import functional as F


class MiniGPT(nn.Module):
    def __init__(self, vocab_size, n_embd=64, block_size=64):
        super().__init__()
        self.block_size = block_size

        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.position_embedding = nn.Embedding(block_size, n_embd)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=n_embd,
            nhead=4,
            dim_feedforward=256,
            batch_first=True
        )

        self.transformer_blocks = nn.TransformerEncoder(encoder_layer, num_layers=3)

        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx):
        B, T = idx.shape

        tok_emb = self.token_embedding(idx)

        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        pos_emb = self.position_embedding(pos)

        x = tok_emb + pos_emb

        mask = nn.Transformer.generate_square_subsequent_mask(T).to(idx.device)

        x = self.transformer_blocks(x, mask=mask, is_causal=True)

        logits = self.lm_head(x)

        return logits


if __name__ == "__main__":
    vocab_size = 65
    fake_input = torch.randint(0, vocab_size, (1, 8))

    model = MiniGPT(vocab_size=vocab_size, block_size=8)
    output = model(fake_input)

    print("=" * 50)
    print("MÓZG MINI-GPT ZBUDOWANY POPRAWNIE!")
    print(f"Wejście: 1 paczka z {fake_input.shape[1]} znakami.")
    print(f"Wyjście: {output.shape} -> [Paczka, Znak, SzansaNaKażdąz65Liter]")
    print("=" * 50)