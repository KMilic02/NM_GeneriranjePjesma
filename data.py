import pandas as pd
import re
import torch
from torch.utils.data import Dataset, DataLoader
from collections import Counter
from sklearn.model_selection import train_test_split

SPECIAL_TOKENS = {
    "<PAD>": 0,
    "<UNK>": 1,
    "<SOS>": 2,
    "<EOS>": 3,
    "<LINE>": 4, # represents the newline symbol at the end of the line
    "<STANZA>": 5, # represents the double newline symbol that marks the beginnning of a new verse
}

def clean_lyrics(text):
    text = re.sub(r"\[.*?\]", " ", text)
    text = re.sub(r"\(.*?\)", " ", text)
    text = text.replace("\r", "\n")
    text = re.sub(r"\n{2,}", " <STANZA> ", text) # new verse break
    text = re.sub(r"\n", " <LINE> ", text) # single line break
    text = re.sub(r"[^a-zA-Z0-9'?!.,<>\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.lower()

class Vocab:
    def __init__(self, frequencies, max_size=-1, min_freq=0):
        self.max_size = max_size
        self.min_freq = min_freq
        self.frequencies = frequencies
        self.stoi = {}
        self.itos = {}

        for token in SPECIAL_TOKENS:
            self._add_token(token)

        for token, freq in sorted(frequencies.items(), key=lambda x: x[1], reverse=True):
            if token in self.stoi:
                continue # already added as special
            if freq < min_freq:
                continue
            if 0 < max_size <= len(self.stoi):
                break
            self._add_token(token)

    def _add_token(self, token):
        idx = len(self.stoi)
        self.stoi[token] = idx
        self.itos[idx] = token

    def __len__(self):
        return len(self.stoi)

    def encode(self, tokens):
        if isinstance(tokens, str):
            return torch.tensor(self.stoi.get(tokens, self.stoi["<UNK>"]))
        return torch.tensor([self.stoi.get(tok, self.stoi["<UNK>"]) for tok in tokens])

class LyricsDataset(Dataset):
    def __init__(self, token_lists, vocab, seq_len=100, stride=1):
        """
        token_lists: list of tokenized lyrics (each already includes <SOS>/<LINE>/<STANZA>/<EOS>)
        vocab: Vocab instance
        seq_len: number of input tokens per sample
        """
        self.seq_len = seq_len
        self.encoded_lyrics = [vocab.encode(tokens) for tokens in token_lists]
        self.index = []
        for song_idx, encoded in enumerate(self.encoded_lyrics):
            max_start = len(encoded) - seq_len - 1
            for start in range(0, max_start + 1, stride):
                self.index.append((song_idx, start))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        song_idx, start = self.index[idx]
        seq = self.encoded_lyrics[song_idx]
        inputs = seq[start : start + self.seq_len]
        targets = seq[start + 1 : start + self.seq_len + 1]
        return inputs, targets

def build_vocab(token_lists, max_size=-1, min_freq=5):
    freqs = Counter()
    for tokens in token_lists:
        freqs.update(tokens)
    return Vocab(freqs, max_size=max_size, min_freq=min_freq)

def df_to_token_lists(frame):
    return [["<SOS>", *lyric.split(), "<EOS>"] for lyric in frame["lyrics"]]

def generate_rap_song_lyrics():
    dataset_csv = pd.read_csv("song_lyrics.csv")

    dataset_csv["year"] = pd.to_numeric(dataset_csv["year"], errors="coerce")

    is_english = (
            dataset_csv["language"].eq("en") |
            (
                    dataset_csv["language"].isna() &
                    dataset_csv["language_ft"].eq("en")
            ) |
            (
                    dataset_csv["language"].isna() &
                    dataset_csv["language_cld3"].eq("en")
            )
    )

    en = dataset_csv[is_english]

    subset = (
        en[en["tag"].isin(["pop"])]
        .dropna(subset=["lyrics", "year"])
        .query("2000 <= year <= 2010")
        .drop_duplicates(subset=["title", "artist"])
        .reset_index(drop=True)
    )

    subset["lyrics"] = subset["lyrics"].map(clean_lyrics)
    subset = subset[subset["lyrics"].str.strip().astype(bool)]
    subset = subset.sample(n=10000, random_state=1337)
    subset.to_csv("pop_song_lyrics.csv", index=False)

def preprocess_data():
    df = pd.read_csv("pop_song_lyrics.csv")

    # TRAIN/VAL/TEST - 80/10/10
    train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

    train_tokens = df_to_token_lists(train_df)
    val_tokens = df_to_token_lists(val_df)
    test_tokens = df_to_token_lists(test_df)

    vocab = build_vocab(train_tokens, min_freq=20)

    freqs = Counter()
    for lyric in df["lyrics"]:
        freqs.update(lyric.split())
    print(len(freqs))

    print(len(vocab))

    unk_id = vocab.stoi["<UNK>"]
    unk_count = total = 0
    for tokens in val_tokens:
        ids = vocab.encode(tokens)
        unk_count += (ids == unk_id).sum().item()
        total += len(ids)
    unk_ratio = unk_count / total
    print(unk_ratio)

    train_dataset = LyricsDataset(train_tokens, vocab, seq_len=100, stride=5)
    val_dataset = LyricsDataset(val_tokens, vocab, seq_len=100, stride=20)
    test_dataset = LyricsDataset(test_tokens, vocab, seq_len=100, stride=20)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    return train_loader, val_loader, test_loader, vocab