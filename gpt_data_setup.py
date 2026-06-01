import torch
from torch.utils.data import Dataset, DataLoader
import os


class ShakespeareDataset(Dataset):
    def __init__(self, text_file, block_size=64):
        print(f"Ładowanie dzieł Szekspira z {text_file}...")
        with open(text_file, 'r', encoding='utf-8') as f:
            self.text = f.read()

        chars = sorted(list(set(self.text)))
        self.vocab_size = len(chars)

        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}

        self.data = [self.stoi[c] for c in self.text]
        self.block_size = block_size

        print(f"Długość tekstu: {len(self.text)} znaków.")
        print(f"Unikalnych znaków (Słownik): {self.vocab_size}")

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        chunk = self.data[idx: idx + self.block_size + 1]

        x = torch.tensor(chunk[:-1], dtype=torch.long)

        y = torch.tensor(chunk[1:], dtype=torch.long)

        return x, y

    def decode(self, list_of_ints):
        return ''.join([self.itos[i] for i in list_of_ints])


if __name__ == "__main__":
    file_path = "shakespeare.txt"

    if not os.path.exists(file_path):
        print(f"Błąd: Nie znaleziono pliku {file_path}!")
    else:
        dataset = ShakespeareDataset(file_path, block_size=8)

        x_test, y_test = dataset[0]

        print("\n" + "=" * 50)
        print("PIERWSZA PRÓBKA DANYCH:")
        print(f"Wejście (X): {x_test.tolist()} -> Tłumaczenie: '{dataset.decode(x_test.tolist())}'")
        print(f"Cel (Y)    : {y_test.tolist()} -> Tłumaczenie: '{dataset.decode(y_test.tolist())}'")
        print("=" * 50)

        print("\nJak sieć będzie się tego uczyć krok po kroku?")
        for t in range(8):
            context = x_test[:t + 1]
            target = y_test[t]
            print(
                f"Gdy maszyna widzi: '{dataset.decode(context.tolist())}' -> Musi zgadnąć: '{dataset.decode([target.item()])}'")