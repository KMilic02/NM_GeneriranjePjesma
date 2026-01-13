import torch, torch.nn as nn
import torch.nn.functional as F
from train import train_epoch, eval_epoch
from model import LyricsLSTM, LyricsGRU
from data import preprocess_data, SPECIAL_TOKENS
import os

def save_checkpoint(model, optimizer, epoch, train_loss, val_loss, path="checkpoints"):
    os.makedirs(path, exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "train_loss": train_loss,
        "val_loss": val_loss,
    }

    filename = "checkpoint.pt"
    torch.save(checkpoint, os.path.join(path, filename))


def load_checkpoint(path, model, optimizer=None, device="cpu"):
    checkpoint = torch.load(path, map_location=device)

    model.load_state_dict(checkpoint["model_state"])

    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state"])

    start_epoch = checkpoint["epoch"]
    return start_epoch

@torch.no_grad()
def generate_lyrics(
    model,
    vocab,
    max_tokens=300,
    temperature=1.0,
    device="cpu"
):
    model.eval()

    idx = torch.tensor([[vocab.stoi["<sos>"]]], device=device)
    hidden = None
    output_tokens = []

    for _ in range(max_tokens):
        logits, hidden = model(idx, hidden)
        logits = logits[:, -1, :] / temperature
        probs = F.softmax(logits, dim=-1)

        next_id = torch.multinomial(probs, 1)
        token = vocab.itos[next_id.item()]

        if token == "<eos>":
            break

        output_tokens.append(token)
        idx = next_id

    text = []
    for tok in output_tokens:
        if tok == "<line>":
            text.append("\n")
        elif tok == "<stanza>":
            text.append("\n\n")
        elif tok == "<unk>":
            pass
        else:
            text.append(tok + " ")

    return "".join(text)

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    #checkpoint = torch.load("checkpoints/checkpoint.pt", map_location=device)

    train_loader, val_loader, test_loader, vocab = preprocess_data()
    vocab_size = len(vocab)
    pad_idx = SPECIAL_TOKENS["<pad>"]

    model_type = "gru"  # Možeš postaviti "gru" ili "lstm", ovisno o tome što odabereš
    if model_type == "lstm":
        model = LyricsLSTM(
            vocab_size=vocab_size,
            embed_dim=128,
            hidden_dim=256,
            num_layers=2,
            dropout=0.3,
            pad_idx=pad_idx
        ).to(device)
    else:
        model = LyricsGRU(
            vocab_size=vocab_size,
            embed_dim=128,
            hidden_dim=256,
            num_layers=2,
            dropout=0.3,
            pad_idx=pad_idx
        ).to(device)

    #model.load_state_dict(checkpoint["model_state"])
    #model.to(device)
    #model.eval()

    #lyrics = generate_lyrics(
    #    model=model,
    #    vocab=vocab,
    #    max_tokens=400,
    #    temperature=0.9,
    #    device=device
    #)

    #print(lyrics)
    #return

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

    epochs = 20
    for epoch in range(epochs):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = eval_epoch(model, val_loader, criterion, device)

        save_checkpoint(
            model, optimizer, epoch + 1,
            train_loss, val_loss,
            path="checkpoints"
        )

        print(f"Epoch {epoch + 1}, Train {train_loss}, Val {val_loss}")

if __name__ == '__main__':
    main()