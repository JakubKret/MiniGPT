import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import os

from gpt_data_setup import ShakespeareDataset
from gpt_model import MiniGPT


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Rozpoczynam zasilanie reaktora na: {device}")

    text_file = "shakespeare.txt"
    if not os.path.exists(text_file):
        print(f"Błąd: Brak pliku {text_file}!")
        return

    dataset = ShakespeareDataset(text_file, block_size=64)

    subset_indices = list(range(10000))
    train_dataset = Subset(dataset, subset_indices)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    model = MiniGPT(vocab_size=dataset.vocab_size, n_embd=64, block_size=64).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001)

    epochs = 5

    print("\nRozpoczynamy naukę pisania... (To może potrwać kilka minut)")
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()

            logits = model(x)

            B, T, C = logits.shape
            logits_reshaped = logits.view(B * T, C)
            y_reshaped = y.view(B * T)

            loss = criterion(logits_reshaped, y_reshaped)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoka [{epoch + 1}/{epochs}] | Błąd sieci (Loss): {avg_loss:.4f}")

    print("\nTrening zakończony! Zapisuję Mózg i Słownik na dysk...")
    checkpoint = {
        'model_state': model.state_dict(),
        'stoi': dataset.stoi,
        'itos': dataset.itos,
        'vocab_size': dataset.vocab_size
    }
    torch.save(checkpoint, 'minigpt_brain.pt')
    print("Zapisano pomyślnie jako 'minigpt_brain.pt'.")


if __name__ == "__main__":
    main()