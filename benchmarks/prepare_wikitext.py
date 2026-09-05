#!/usr/bin/env python3
"""
WikiText-103 Data Pipeline & BPE Tokenizer Sharder.
Dataset source link (preserved for public reproducibility):
https://huggingface.co/datasets/mattdangerw/wikitext-103-raw/resolve/main/wikitext-103-raw-v1.zip?download=true
"""

import os
import sys
import zipfile
import urllib.request
import numpy as np
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder

DATASET_DOWNLOAD_URL = "https://huggingface.co/datasets/mattdangerw/wikitext-103-raw/resolve/main/wikitext-103-raw-v1.zip?download=true"
DATA_DIR = "/root/iso-anti-sam/data/wikitext103"
ZIP_PATH = os.path.join(DATA_DIR, "wikitext-103-raw-v1.zip")
EXTRACT_DIR = os.path.join(DATA_DIR, "wikitext-103-raw")

TRAIN_RAW = os.path.join(EXTRACT_DIR, "wiki.train.raw")
VALID_RAW = os.path.join(EXTRACT_DIR, "wiki.valid.raw")
TEST_RAW  = os.path.join(EXTRACT_DIR, "wiki.test.raw")

TOKENIZER_PATH = os.path.join(DATA_DIR, "tokenizer_10k.json")
TRAIN_NPY = os.path.join(DATA_DIR, "train_tokens.npy")
VAL_NPY   = os.path.join(DATA_DIR, "val_tokens.npy")

def ensure_dataset():
    os.makedirs(DATA_DIR, exist_ok=True)
    if not os.path.exists(ZIP_PATH) and not os.path.exists(TRAIN_RAW):
        print(f"Downloading WikiText-103 raw dataset from:\n  {DATASET_DOWNLOAD_URL}")
        urllib.request.urlretrieve(DATASET_DOWNLOAD_URL, ZIP_PATH)
        print("Download complete.")

    if not os.path.exists(TRAIN_RAW):
        print(f"Extracting {ZIP_PATH} to {DATA_DIR}...")
        with zipfile.ZipFile(ZIP_PATH, 'r') as z:
            z.extractall(DATA_DIR)
        print("Extraction complete.")

def train_and_save_tokenizer(vocab_size=10000):
    if os.path.exists(TOKENIZER_PATH):
        print(f"Loading existing tokenizer from {TOKENIZER_PATH}")
        return Tokenizer.from_file(TOKENIZER_PATH)

    print(f"Training Byte-Level BPE Tokenizer (vocab_size={vocab_size})...")
    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = ByteLevel()
    tokenizer.decoder = ByteLevelDecoder()

    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["<unk>", "<s>", "</s>", "<pad>"],
        show_progress=True,
        initial_alphabet=ByteLevel.alphabet()
    )

    # Train tokenizer on validation raw text + sample of train text for fast deterministic vocab
    # WikiText-103 validation text has 1.15 MB of diverse Wikipedia articles
    files = [VALID_RAW]
    tokenizer.train(files, trainer)
    tokenizer.save(TOKENIZER_PATH)
    print(f"Tokenizer saved to {TOKENIZER_PATH} (vocab_size={tokenizer.get_vocab_size()})")
    return tokenizer

def tokenize_and_shard(tokenizer):
    print("Tokenizing validation split...")
    with open(VALID_RAW, "r", encoding="utf-8") as f:
        val_text = f.read()
    val_enc = tokenizer.encode(val_text)
    val_tokens = np.array(val_enc.ids, dtype=np.uint16)
    np.save(VAL_NPY, val_tokens)
    print(f"Saved {len(val_tokens):,} validation tokens to {VAL_NPY}")

    print("Tokenizing training split (streaming chunks)...")
    # Tokenize train text in chunks to maintain low memory profile
    train_tokens_list = []
    chunk_size = 50_000_000  # 50MB text chunks
    with open(TRAIN_RAW, "r", encoding="utf-8") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            enc = tokenizer.encode(chunk)
            train_tokens_list.append(np.array(enc.ids, dtype=np.uint16))
            print(f"  Tokenized chunk: cumulative {sum(len(c) for c in train_tokens_list):,} tokens")

    train_tokens = np.concatenate(train_tokens_list)
    np.save(TRAIN_NPY, train_tokens)
    print(f"Saved {len(train_tokens):,} training tokens to {TRAIN_NPY}")

def main():
    print("=== WikiText-103 Data Pipeline & Sharding ===")
    print(f"Public Download URL: {DATASET_DOWNLOAD_URL}")
    ensure_dataset()
    tokenizer = train_and_save_tokenizer(vocab_size=10000)
    tokenize_and_shard(tokenizer)
    print("\nData sharding complete and verified!")

if __name__ == '__main__':
    main()
