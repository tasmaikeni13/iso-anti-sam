#!/usr/bin/env python3
"""
WikiText-103 Autoregressive Language Modeling Benchmark on AMD Instinct MI300X.
Optimized with:
- PyTorch FlashAttention / Scaled Dot-Product Attention (SDPA) hardware acceleration.
- Native IsoAntiSAM with Bilateral Coherence Gating and Isochoric Gauging.
- Hessian spectral sharpness (lambda_max) power iteration diagnostics.
- Baseline comparisons: AdamW, SGD with momentum, Standard SAM, Raw Anti-SAM, IsoAntiSAM.
"""

import os
import sys
import time
import math
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Ensure package is importable
sys.path.insert(0, '/root/iso-anti-sam/src')
from iso_anti_sam import AntiSAM, IsoAntiSAM

# -----------------------------------------------------------------------------
# 1. Model Architecture: 6-layer Causal Transformer with FlashAttention
# -----------------------------------------------------------------------------
class FlashCausalSelfAttention(nn.Module):
    def __init__(self, dim=384, n_heads=6, seq_len=256, dropout=0.1):
        super().__init__()
        assert dim % n_heads == 0
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.dropout_p = dropout

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        drop_p = self.dropout_p if self.training else 0.0
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=drop_p)
        y = y.permute(0, 2, 1, 3).reshape(B, T, C)
        return self.proj(y)

class MLP(nn.Module):
    def __init__(self, dim=384, hidden_dim=1536, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.fc2(self.act(self.fc1(x))))

class TransformerBlock(nn.Module):
    def __init__(self, dim=384, n_heads=6, seq_len=256, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = FlashCausalSelfAttention(dim, n_heads, seq_len, dropout)
        self.ln2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, 4 * dim, dropout)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class CausalTransformer(nn.Module):
    def __init__(self, vocab_size=10000, dim=384, n_heads=6, n_layers=6, seq_len=256, dropout=0.1):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, dim)
        self.pos_emb = nn.Embedding(seq_len, dim)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            TransformerBlock(dim, n_heads, seq_len, dropout) for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight  # Weight tying (~14.59M parameters)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def forward(self, idx):
        B, T = idx.shape
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device).unsqueeze(0)
        x = self.drop(self.token_emb(idx) + self.pos_emb(pos))
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits

# -----------------------------------------------------------------------------
# 2. Dataset Loader
# -----------------------------------------------------------------------------
class TokenizedWikiDataset(Dataset):
    def __init__(self, npy_path, seq_len=256):
        self.data = np.load(npy_path, mmap_mode='r')
        self.seq_len = seq_len
        self.n_samples = (len(self.data) - 1) // seq_len

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        start = idx * self.seq_len
        end = start + self.seq_len + 1
        chunk = torch.from_numpy(self.data[start:end].astype(np.int64))
        x = chunk[:-1]
        y = chunk[1:]
        return x, y

# Standard SAM implementation
class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer_cls=torch.optim.AdamW, rho=0.05, **kwargs):
        defaults = dict(rho=rho, **kwargs)
        super().__init__(params, defaults)
        self.base_optimizer = base_optimizer_cls(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)
            for p in group["params"]:
                if p.grad is None:
                    continue
                self.state[p]["old_p"] = p.data.clone()
                p.add_(p.grad * scale)
        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if "old_p" in self.state[p]:
                    p.data.copy_(self.state[p]["old_p"])
                    del self.state[p]["old_p"]
        self.base_optimizer.step()
        if zero_grad:
            self.zero_grad()

    def zero_grad(self, set_to_none=False):
        self.base_optimizer.zero_grad(set_to_none=set_to_none)

    def _grad_norm(self):
        device = self.param_groups[0]["params"][0].device
        norms = [
            torch.linalg.vector_norm(p.grad, 2).to(device)
            for group in self.param_groups for p in group["params"]
            if p.grad is not None
        ]
        if not norms:
            return torch.tensor(0.0, device=device)
        return torch.linalg.vector_norm(torch.stack(norms), 2)

# -----------------------------------------------------------------------------
# 3. Hessian Sharpness Diagnostic (Power Iteration for lambda_max)
# -----------------------------------------------------------------------------
def compute_hessian_max_eigenvalue(model, x, y, criterion, n_iters=3, r=1e-4):
    params = [p for p in model.parameters() if p.requires_grad]
    v = [torch.randn_like(p) for p in params]
    v_norm = torch.sqrt(sum((p**2).sum() for p in v))
    v = [p / (v_norm + 1e-12) for p in v]

    model.zero_grad()
    loss = criterion(model(x).view(-1, 10000), y.view(-1))
    loss.backward()
    g_base = [p.grad.clone() for p in params]
    model.zero_grad()

    lambda_max = 0.0
    for _ in range(n_iters):
        with torch.no_grad():
            for p, vi in zip(params, v):
                p.add_(vi * r)
        loss_pert = criterion(model(x).view(-1, 10000), y.view(-1))
        loss_pert.backward()
        g_pert = [p.grad.clone() for p in params]
        model.zero_grad()

        with torch.no_grad():
            for p, vi in zip(params, v):
                p.sub_(vi * r)

        Hv = [(gp - gb) / r for gp, gb in zip(g_pert, g_base)]
        lambda_max = sum((vi * hvi).sum().item() for vi, hvi in zip(v, Hv))
        Hv_norm = torch.sqrt(sum((hvi**2).sum() for hvi in Hv))
        v = [hvi / (Hv_norm + 1e-12) for hvi in Hv]

    return float(max(0.0, lambda_max))

# -----------------------------------------------------------------------------
# 4. Main Benchmark Driver
# -----------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="WikiText-103 Benchmark on AMD MI300X")
    parser.add_argument('--optimizer', type=str, default='adamw',
                        choices=['adamw', 'sgd', 'sam', 'anti_sam', 'iso_anti_sam'])
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--steps_per_epoch', type=int, default=200)
    parser.add_argument('--val_steps', type=int, default=50)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--rho', type=float, default=0.05)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--seq_len', type=int, default=256)
    parser.add_argument('--dim', type=int, default=384)
    parser.add_argument('--n_layers', type=int, default=6)
    parser.add_argument('--n_heads', type=int, default=6)
    parser.add_argument('--device', type=str, default='cuda:0' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--exp_name', type=str, default=None)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--log_dir', type=str, default='/root/iso-anti-sam/logs/wikitext')
    return parser.parse_args()

def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.log_dir, exist_ok=True)

    device = torch.device(args.device)
    print(f"=== Starting Benchmark: {args.optimizer.upper()} on {device} ===")
    if 'cuda' in str(device):
        print(f"GPU Device: {torch.cuda.get_device_name(0)}")
        print(f"Total VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    if args.lr is None:
        if args.optimizer == 'sgd':
            args.lr = 0.05
        else:
            args.lr = 5e-4

    train_npy = '/root/iso-anti-sam/data/wikitext103/train_tokens.npy'
    val_npy = '/root/iso-anti-sam/data/wikitext103/val_tokens.npy'

    assert os.path.exists(train_npy) and os.path.exists(val_npy), "Tokens not found in data directory"
    train_dataset = TokenizedWikiDataset(train_npy, seq_len=args.seq_len)
    val_dataset = TokenizedWikiDataset(val_npy, seq_len=args.seq_len)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, drop_last=True)

    model = CausalTransformer(vocab_size=10000, dim=args.dim, n_heads=args.n_heads, n_layers=args.n_layers, seq_len=args.seq_len).to(device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {num_params:,} ({num_params/1e6:.2f}M) with FlashAttention SDPA")

    if args.optimizer == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    elif args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
    elif args.optimizer == 'sam':
        optimizer = SAM(model.parameters(), base_optimizer_cls=torch.optim.AdamW, rho=args.rho, lr=args.lr, weight_decay=0.01)
    elif args.optimizer == 'anti_sam':
        optimizer = AntiSAM(model.parameters(), base_optimizer_cls=torch.optim.AdamW, rho=args.rho, lr=args.lr, weight_decay=0.01)
    elif args.optimizer == 'iso_anti_sam':
        optimizer = IsoAntiSAM(model.parameters(), base_optimizer_cls=torch.optim.AdamW, rho=args.rho, lr=args.lr, weight_decay=0.01)

    total_steps = args.epochs * args.steps_per_epoch
    warmup_steps = min(50, total_steps // 10)
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))

    base_opt = optimizer.base_optimizer if hasattr(optimizer, 'base_optimizer') else optimizer
    scheduler = torch.optim.lr_scheduler.LambdaLR(base_opt, lr_lambda)

    criterion = nn.CrossEntropyLoss()
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_ppl': [],
        'generalization_gap': [],
        'epoch_time_s': [],
        'lambda_max': []
    }

    print(f"Configuration: lr={args.lr}, rho={args.rho}, batch_size={args.batch_size}, epochs={args.epochs}, steps/epoch={args.steps_per_epoch}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_start = time.time()
        total_loss = 0.0
        step_count = 0

        for step, (x, y) in enumerate(train_loader):
            if step >= args.steps_per_epoch:
                break
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

            if args.optimizer in ['adamw', 'sgd']:
                logits = model(x)
                loss = criterion(logits.view(-1, 10000), y.view(-1))
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                total_loss += loss.item()

            elif args.optimizer in ['anti_sam', 'sam']:
                logits = model(x)
                loss = criterion(logits.view(-1, 10000), y.view(-1))
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.first_step(zero_grad=True)

                logits2 = model(x)
                loss2 = criterion(logits2.view(-1, 10000), y.view(-1))
                loss2.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.second_step(zero_grad=True)
                total_loss += loss.item()

            elif args.optimizer == 'iso_anti_sam':
                # Micro-batch 1
                half_b = args.batch_size // 2
                x1, y1 = x[:half_b], y[:half_b]
                logits1 = model(x1)
                l1 = criterion(logits1.view(-1, 10000), y1.view(-1))
                optimizer.zero_grad()
                l1.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                g1 = [p.grad.clone() if p.grad is not None else None for p in model.parameters()]

                # Micro-batch 2
                x2, y2 = x[half_b:], y[half_b:]
                logits2 = model(x2)
                l2 = criterion(logits2.view(-1, 10000), y2.view(-1))
                optimizer.zero_grad()
                l2.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                g2 = [p.grad.clone() if p.grad is not None else None for p in model.parameters()]

                optimizer.zero_grad()
                cos_sim = optimizer.compute_bilateral_perturbation(g1, g2)

                logits_full = model(x)
                loss_full = criterion(logits_full.view(-1, 10000), y.view(-1))
                loss_full.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step_with_bilateral(zero_grad=True)
                total_loss += loss_full.item()

            scheduler.step()
            step_count += 1

        avg_train_loss = total_loss / max(1, step_count)
        epoch_dur = time.time() - epoch_start

        # Validation
        model.eval()
        val_loss_total = 0.0
        val_count = 0
        with torch.no_grad():
            for v_step, (vx, vy) in enumerate(val_loader):
                if v_step >= args.val_steps:
                    break
                vx, vy = vx.to(device, non_blocking=True), vy.to(device, non_blocking=True)
                v_logits = model(vx)
                v_loss = criterion(v_logits.view(-1, 10000), vy.view(-1))
                val_loss_total += v_loss.item()
                val_count += 1

        avg_val_loss = val_loss_total / max(1, val_count)
        val_ppl = math.exp(min(avg_val_loss, 20.0))
        gen_gap = abs(avg_val_loss - avg_train_loss)

        sample_vx, sample_vy = next(iter(val_loader))
        sample_vx, sample_vy = sample_vx.to(device), sample_vy.to(device)
        lambda_max = compute_hessian_max_eigenvalue(model, sample_vx, sample_vy, criterion, n_iters=3)

        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['val_ppl'].append(val_ppl)
        history['generalization_gap'].append(gen_gap)
        history['epoch_time_s'].append(epoch_dur)
        history['lambda_max'].append(lambda_max)

        print(f"Epoch {epoch:2d}/{args.epochs:2d} | Train: {avg_train_loss:.4f} | Val: {avg_val_loss:.4f} | Gap: {gen_gap:.4f} | PPL: {val_ppl:6.2f} | λ_max: {lambda_max:6.2f} | Time: {epoch_dur:.2f}s")

    exp_prefix = args.exp_name if args.exp_name else args.optimizer
    log_file = os.path.join(args.log_dir, f"{exp_prefix}_history.json")
    with open(log_file, 'w') as f:
        json.dump(history, f, indent=2)

    ckpt_file = os.path.join(args.log_dir, f"{exp_prefix}_checkpoint.pt")
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'history': history,
        'config': vars(args)
    }, ckpt_file)

    print(f"\n--- Benchmark Complete: {args.optimizer.upper()} ({exp_prefix}) ---")
    print(f"Final Val Loss: {history['val_loss'][-1]:.4f} | Final Val PPL: {history['val_ppl'][-1]:.2f} | Gap: {history['generalization_gap'][-1]:.4f} | λ_max: {history['lambda_max'][-1]:.2f}")
    print(f"Logs saved to: {log_file}")

if __name__ == '__main__':
    main()
