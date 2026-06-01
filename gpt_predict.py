import torch
import torch.nn.functional as F
import os

from gpt_model import MiniGPT


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_file = 'minigpt_brain.pt'

    if not os.path.exists(checkpoint_file):
        print(f"Błąd: Nie znaleziono Mózgu w {checkpoint_file}!")
        return

    checkpoint = torch.load(checkpoint_file, map_location=device, weights_only=False)
    stoi = checkpoint['stoi']
    itos = checkpoint['itos']
    vocab_size = checkpoint['vocab_size']

    model = MiniGPT(vocab_size=vocab_size, n_embd=64, block_size=64).to(device)
    model.load_state_dict(checkpoint['model_state'])
    model.eval()

    def encode(text):
        return [stoi[c] for c in text if c in stoi]

    def decode(lista_int):
        return ''.join([itos[i] for i in lista_int])

    print("\n" + "=" * 50)
    print("🎭 GENERATOR SZEKSPIRA (MINI-GPT) GOTOWY")
    print("Wpisz początek zdania (np. 'ROMEO:' lub 'O, my lord').")
    print("Wpisz 'exit', aby wyjść.")
    print("=" * 50 + "\n")

    block_size = 64

    while True:
        prompt = input("\nPodaj początek tekstu: ")

        if prompt.lower() == 'exit':
            break
        if not prompt:
            continue

        context = torch.tensor(encode(prompt), dtype=torch.long, device=device).unsqueeze(0)

        if context.shape[1] == 0:
            print("Użyłeś znaków, których Szekspir nie znał. Spróbuj czystego angielskiego!")
            continue

        print("\nGeneruję...")
        print("-" * 40)

        print(prompt, end="")

        max_new_tokens = 300

        with torch.no_grad():
            for _ in range(max_new_tokens):
                context_cropped = context[:, -block_size:]

                logits = model(context_cropped)
                logits_last_char = logits[:, -1, :]

                probs = F.softmax(logits_last_char, dim=-1)

                next_char_idx = torch.multinomial(probs, num_samples=1)

                print(decode([next_char_idx.item()]), end="", flush=True)

                context = torch.cat((context, next_char_idx), dim=1)

        print("\n" + "-" * 40)


if __name__ == "__main__":
    main()