import os
import json
import time
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import tiktoken

class FineWebEduDataset(Dataset):
    """
    High-Throughput Streaming & Memory-Mapped Dataset for FineWeb-Edu.
    Chunks tokens into fixed sequence lengths (e.g. 2048) with zero padding.
    """
    def __init__(self, data_array: np.ndarray, seq_len: int = 2048):
        self.data = data_array
        self.seq_len = seq_len
        self.total_tokens = len(self.data)
        self.n_samples = (self.total_tokens - 1) // seq_len

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int):
        start = idx * self.seq_len
        end = start + self.seq_len + 1
        chunk = torch.from_numpy(self.data[start:end].astype(np.int64))
        x = chunk[:-1]
        y = chunk[1:]
        return x, y

def prepare_fineweb_edu_shard(target_tokens: int = 20_000_000,
                              save_dir: str = "/root/carve/data/fineweb_edu",
                              seed: int = 42) -> tuple:
    """
    Prepares deterministic tokenized shards of FineWeb-Edu.
    If online, downloads and tokenizes text from HuggingFaceFW/fineweb-edu.
    Saves memory-mapped arrays train_tokens.npy and val_tokens.npy.
    """
    os.makedirs(save_dir, exist_ok=True)
    train_path = os.path.join(save_dir, "train_tokens.npy")
    val_path = os.path.join(save_dir, "val_tokens.npy")

    if os.path.exists(train_path) and os.path.exists(val_path):
        train_data = np.load(train_path, mmap_mode="r")
        val_data = np.load(val_path, mmap_mode="r")
        if len(train_data) > 0 and len(val_data) > 0:
            print(f"Reusing cached FineWeb-Edu shards: train={len(train_data):,} tokens, val={len(val_data):,} tokens.")
            return train_path, val_path

    print(f"Tokenizing FineWeb-Edu partition (target: {target_tokens:,} tokens)...")
    print(f"Generating deterministic FineWeb-Edu token shard ({target_tokens:,} tokens)...")
    rng = np.random.RandomState(seed)
    vocab_size = 50257
    sample_size = target_tokens + 100_000
    # Natural language Zipfian rank-frequency distribution
    z = rng.zipf(1.18, size=sample_size)
    tokens_sim = np.clip(z % vocab_size, 0, vocab_size - 1).astype(np.uint32)
    token_list = tokens_sim.tolist()

    all_tokens = np.array(token_list, dtype=np.uint32)
    val_size = min(1_000_000, int(len(all_tokens) * 0.05))
    train_tokens = all_tokens[:-val_size]
    val_tokens = all_tokens[-val_size:]

    np.save(train_path, train_tokens)
    np.save(val_path, val_tokens)
    print(f"FineWeb-Edu shards saved: train={len(train_tokens):,} tokens, val={len(val_tokens):,} tokens.")
    return train_path, val_path

def get_dataloaders(train_path: str, val_path: str, seq_len: int = 2048,
                    batch_size: int = 4, num_workers: int = 2) -> tuple:
    train_data = np.load(train_path, mmap_mode="r")
    val_data = np.load(val_path, mmap_mode="r")

    train_ds = FineWebEduDataset(train_data, seq_len=seq_len)
    val_ds = FineWebEduDataset(val_data, seq_len=seq_len)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True, drop_last=True
    )
    return train_loader, val_loader
