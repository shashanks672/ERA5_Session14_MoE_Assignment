"""
Dataset and Tokenizer for Language Model Pre-training.

Provides:
- Lightweight, self-contained Character-level Tokenizer
- Tiny Shakespeare dataset downloader with automatic offline fallback
- PyTorch Dataset and DataLoader generating causal LM pairs (x, y)
"""

import os
import urllib.request
from typing import Tuple, Dict, List, Optional
import torch
from torch.utils.data import Dataset, DataLoader


SAMPLE_FALLBACK_TEXT = """First Citizen:
Before we proceed any further, hear me speak.

All:
Speak, speak.

First Citizen:
You are all resolved rather to die than to famish?

All:
Resolved. resolved.

First Citizen:
First, you know Caius Marcius is chief enemy to the people.

All:
We know't, we know't.

First Citizen:
Let us kill him, and we'll have corn at our own price.
Is't a verdict?

All:
No more talking on't; let it be done: away, away!

Second Citizen:
One word, good citizens.

First Citizen:
We are accounted poor citizens, the patricians good.
What authority surfeits on would relieve us: if they
would yield us but the superfluity, while it were
wholesome, we might guess they relieved us humanely;
but they think we are too dear: the leanness that
afflicts us, the object of our misery, is as an
inventory to particularise their abundance; our
sufferance is a gain to them. Let us revenge this with
our pikes, ere we become rakes: for the gods know I
speak this in hunger for bread, not in thirst for revenge.

Second Citizen:
Would you proceed especially against Caius Marcius?

All:
Against him first: he's a very dog to the commonalty.

Second Citizen:
Consider you what services he has done for his country?

First Citizen:
Very well; and could be content to give him good
report for't, but that he pays himself with being proud.

Second Citizen:
Nay, but speak not maliciously.

First Citizen:
I say unto you, what he hath done famously, he did
it to that end: though soft-conscienced men can be
content to say it was for his country, he did it to
please his mother and to be partly proud; which he
will still be, even to the altitude of his virtue.
""" * 50  # Replicated for sufficient tokens when offline


class CharTokenizer:
    """Simple, transparent character-level tokenizer."""
    def __init__(self, text: Optional[str] = None, vocab: Optional[List[str]] = None):
        if vocab is not None:
            self.chars = sorted(list(set(vocab)))
        elif text is not None:
            self.chars = sorted(list(set(text)))
        else:
            # Default to printable ASCII
            self.chars = [chr(i) for i in range(128)]
            
        self.vocab_size = len(self.chars)
        self.char_to_idx = {ch: i for i, ch in enumerate(self.chars)}
        self.idx_to_char = {i: ch for i, ch in enumerate(self.chars)}

    def encode(self, s: str) -> List[int]:
        """Encodes string to list of token IDs (handles unknown chars by fallback modulo or nearest)."""
        return [self.char_to_idx.get(c, 0) for c in s]

    def decode(self, indices: List[int]) -> str:
        """Decodes list of token IDs back to string."""
        return "".join([self.idx_to_char.get(i, "") for i in indices])


class TextDataset(Dataset):
    """
    Causal Language Modeling Dataset.
    
    Given a 1D tensor of token IDs, returns pairs:
    x = tokens[i : i + seq_len]
    y = tokens[i + 1 : i + seq_len + 1]
    """
    def __init__(self, data: torch.Tensor, seq_len: int):
        self.data = data
        self.seq_len = seq_len
        self.num_samples = len(data) - seq_len

    def __len__(self) -> int:
        return max(0, self.num_samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        chunk = self.data[idx : idx + self.seq_len + 1]
        x = chunk[:-1]
        y = chunk[1:]
        return x, y


def load_dataset(
    data_dir: str = "./data",
    seq_len: int = 128,
    train_split: float = 0.9,
    url: str = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
) -> Tuple[TextDataset, TextDataset, CharTokenizer]:
    """
    Downloads and loads Tiny Shakespeare dataset (or falls back to sample text).
    
    Returns:
        train_dataset: TextDataset for training
        val_dataset: TextDataset for validation
        tokenizer: Fitted CharTokenizer
    """
    os.makedirs(data_dir, exist_ok=True)
    file_path = os.path.join(data_dir, "input.txt")
    
    text = None
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
    else:
        try:
            print(f"Downloading dataset from {url} ...")
            urllib.request.urlretrieve(url, file_path)
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
            print(f"Dataset downloaded successfully ({len(text)} characters).")
        except Exception as e:
            print(f"Warning: Could not download dataset ({e}). Using embedded fallback dataset.")
            text = SAMPLE_FALLBACK_TEXT
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(text)
                
    tokenizer = CharTokenizer(text=text)
    encoded_data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    
    # Train / Val Split
    n_train = int(len(encoded_data) * train_split)
    train_data = encoded_data[:n_train]
    val_data = encoded_data[n_train:]
    
    train_dataset = TextDataset(train_data, seq_len=seq_len)
    val_dataset = TextDataset(val_data, seq_len=seq_len)
    
    return train_dataset, val_dataset, tokenizer


def get_dataloader(
    dataset: TextDataset,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0
) -> DataLoader:
    """Helper to instantiate DataLoader."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=True,
        pin_memory=True if torch.cuda.is_available() else False
    )
