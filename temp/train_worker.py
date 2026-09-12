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

sys.path.insert(0, "/root/carve")
sys.path.insert(0, "/root/carve/src")
from carve import Carve, SAM
from temp.model import CausalLMTransformer125M

class MmapTokenDataset(Dataset):
    def __init__(self, data_path: str, seq_len: int = 2048):
        self.data = np.load(data_path, mmap_mode="r")
        self.seq_len = seq_len
        self.n_samples = (len(self.data) - 1) // seq_len

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int):
        start = idx * self.seq_len
        end = start + self.seq_len + 1
        chunk = torch.from_numpy(self.data[start:end].astype(np.int64))
        return chunk[:-1], chunk[1:]

def get_cosine_lr(step, warmup_steps, max_steps, base_lr, min_lr_ratio=0.1):
    if step < warmup_steps:
        return base_lr * float(step) / float(max(1, warmup_steps))
    progress = float(step - warmup_steps) / float(max(1, max_steps - warmup_steps))
    return base_lr * (min_lr_ratio + 0.5 * (1.0 - min_lr_ratio) * (1.0 + math.cos(math.pi * progress)))

def get_rho_schedule(step, max_steps, rho_0):
    progress = min(1.0, float(step) / float(max(1, max_steps)))
    return rho_0 * 0.5 * (1.0 + math.cos(math.pi * progress))

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--optimizer", type=str, required=True, choices=["adamw", "sam", "carve"])
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--total_tokens", type=int, default=300_000_000)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum_steps", type=int, default=8)
    parser.add_argument("--seq_len", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=6e-4)
    parser.add_argument("--rho", type=float, default=0.02)
    parser.add_argument("--coherence_floor", type=float, default=0.0)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--log_file", type=str, required=True)
    parser.add_argument("--val_interval", type=int, default=100)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--device", type=str, default="cuda:0")
    return parser.parse_args()

def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(os.path.dirname(args.log_file), exist_ok=True)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[{args.optimizer.upper()} Seed {args.seed}] Initializing on {device}...", flush=True)

    # Data
    train_path = "/root/carve/data/fineweb_edu_300m/train_tokens.npy"
    val_path = "/root/carve/data/fineweb_edu_300m/val_tokens.npy"
    assert os.path.exists(train_path) and os.path.exists(val_path), "Data shards not found!"

    train_ds = MmapTokenDataset(train_path, seq_len=args.seq_len)
    val_ds = MmapTokenDataset(val_path, seq_len=args.seq_len)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=0, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=0, pin_memory=True, drop_last=True
    )
    train_iter = iter(train_loader)

    # Model
    model = CausalLMTransformer125M(vocab_size=50257, max_seq_len=args.seq_len).to(device)
    print(f"[{args.optimizer.upper()} Seed {args.seed}] Model parameters: {model.get_num_params():,}", flush=True)

    # Optimizer
    if args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
            betas=(0.9, 0.95), eps=1e-8, fused=torch.cuda.is_available()
        )
    elif args.optimizer == "sam":
        optimizer = SAM(
            model.parameters(), base_optimizer_cls=torch.optim.AdamW,
            lr=args.lr, rho=args.rho, weight_decay=args.weight_decay,
            betas=(0.9, 0.95), eps=1e-8
        )
    elif args.optimizer == "carve":
        optimizer = Carve(
            model.parameters(), base_optimizer_cls=torch.optim.AdamW,
            lr=args.lr, rho=args.rho, coherence_floor=args.coherence_floor,
            weight_decay=args.weight_decay, betas=(0.9, 0.95), eps=1e-8
        )

    tokens_per_step = args.batch_size * args.seq_len * args.grad_accum_steps
    total_steps = (args.total_tokens + tokens_per_step - 1) // tokens_per_step
    warmup_steps = int(total_steps * 0.03)

    print(f"[{args.optimizer.upper()} Seed {args.seed}] Step budget: {total_steps} steps ({tokens_per_step:,} tok/step)", flush=True)

    history = {
        "config": vars(args),
        "total_steps": total_steps,
        "tokens_per_step": tokens_per_step,
        "step": [],
        "tokens_seen": [],
        "train_loss": [],
        "val_loss": [],
        "val_ppl": [],
        "learning_rate": [],
        "rho": [],
        "coherence_cos": [],
        "step_time_s": [],
        "throughput_tokens_per_s": [],
        "wallclock_elapsed_s": []
    }

    start_wallclock = time.time()
    tokens_seen = 0

    for step in range(1, total_steps + 1):
        step_start = time.time()
        curr_lr = get_cosine_lr(step, warmup_steps, total_steps, args.lr)
        for pg in optimizer.param_groups:
            pg["lr"] = curr_lr

        curr_rho = get_rho_schedule(step, total_steps, args.rho)

        model.train()
        accum_loss = 0.0
        coherence_val = 0.0

        batches = []
        for _ in range(args.grad_accum_steps):
            try:
                bx, by = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                bx, by = next(train_iter)
            batches.append((bx.to(device, non_blocking=True), by.to(device, non_blocking=True)))

        if args.optimizer == "carve":
            optimizer.zero_grad(set_to_none=True)
            half = max(1, len(batches) // 2)
            batches1 = batches[:half]
            batches2 = batches[half:]

            # Micro-batch cohort 1
            for bx, by in batches1:
                with torch.amp.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu", dtype=torch.bfloat16):
                    out1 = model(bx)
                    loss1 = F.cross_entropy(out1.view(-1, 50257), by.view(-1)) / len(batches1)
                loss1.backward()
            g1 = [p.grad.clone() if p.grad is not None else None for p in model.parameters()]

            # Micro-batch cohort 2
            model.zero_grad(set_to_none=True)
            for bx, by in batches2:
                with torch.amp.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu", dtype=torch.bfloat16):
                    out2 = model(bx)
                    loss2 = F.cross_entropy(out2.view(-1, 50257), by.view(-1)) / len(batches2)
                loss2.backward()
            g2 = [p.grad.clone() if p.grad is not None else None for p in model.parameters()]

            # Isochoric perturbation
            model.zero_grad(set_to_none=True)
            coherence_val = optimizer.compute_bilateral_perturbation(g1, g2, rho=curr_rho)

            # Outer evaluation on full batch at perturbed weights
            for bx, by in batches:
                with torch.amp.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu", dtype=torch.bfloat16):
                    out_full = model(bx)
                    loss_full = F.cross_entropy(out_full.view(-1, 50257), by.view(-1)) / len(batches)
                loss_full.backward()
                accum_loss += loss_full.item()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step_with_bilateral(zero_grad=True)

        elif args.optimizer == "sam":
            optimizer.zero_grad(set_to_none=True)
            for bx, by in batches:
                with torch.amp.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu", dtype=torch.bfloat16):
                    out1 = model(bx)
                    loss1 = F.cross_entropy(out1.view(-1, 50257), by.view(-1)) / len(batches)
                loss1.backward()
            optimizer.first_step(zero_grad=True)

            for bx, by in batches:
                with torch.amp.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu", dtype=torch.bfloat16):
                    out2 = model(bx)
                    loss2 = F.cross_entropy(out2.view(-1, 50257), by.view(-1)) / len(batches)
                loss2.backward()
                accum_loss += loss2.item()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.second_step(zero_grad=True)

        elif args.optimizer == "adamw":
            optimizer.zero_grad(set_to_none=True)
            for bx, by in batches:
                with torch.amp.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu", dtype=torch.bfloat16):
                    out = model(bx)
                    loss = F.cross_entropy(out.view(-1, 50257), by.view(-1)) / len(batches)
                loss.backward()
                accum_loss += loss.item()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        step_dur = time.time() - step_start
        tokens_seen += tokens_per_step
        throughput = tokens_per_step / max(1e-4, step_dur)
        wallclock = time.time() - start_wallclock

        # Validation
        val_loss = None
        val_ppl = None
        if step % args.val_interval == 0 or step == total_steps:
            model.eval()
            val_loss_tot = 0.0
            val_steps = 5
            with torch.no_grad():
                for v_step, (vx, vy) in enumerate(val_loader):
                    if v_step >= val_steps:
                        break
                    vx, vy = vx.to(device, non_blocking=True), vy.to(device, non_blocking=True)
                    with torch.amp.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu", dtype=torch.bfloat16):
                        vout = model(vx)
                        vloss = F.cross_entropy(vout.view(-1, 50257), vy.view(-1))
                    val_loss_tot += vloss.item()
            val_loss = val_loss_tot / max(1, val_steps)
            val_ppl = math.exp(min(val_loss, 20.0))

        if step == 1 or step % args.log_interval == 0 or step == total_steps:
            v_str = f"Val: {val_loss:.4f} | PPL: {val_ppl:.2f}" if val_loss is not None else "Val: --"
            print(f"[{args.optimizer.upper()} S{args.seed}] Step {step:4d}/{total_steps} ({tokens_seen/1e6:.1f}M tok) | Train: {accum_loss:.4f} | {v_str} | {throughput:,.0f} tok/s | Elapsed: {wallclock:.1f}s", flush=True)

        history["step"].append(step)
        history["tokens_seen"].append(tokens_seen)
        history["train_loss"].append(accum_loss)
        history["val_loss"].append(val_loss if val_loss is not None else (history["val_loss"][-1] if history["val_loss"] else accum_loss))
        history["val_ppl"].append(val_ppl if val_ppl is not None else (history["val_ppl"][-1] if history["val_ppl"] else math.exp(min(accum_loss, 20.0))))
        history["learning_rate"].append(curr_lr)
        history["rho"].append(curr_rho)
        history["coherence_cos"].append(coherence_val)
        history["step_time_s"].append(step_dur)
        history["throughput_tokens_per_s"].append(throughput)
        history["wallclock_elapsed_s"].append(wallclock)

        if step == 1 or step % 10 == 0 or step == total_steps:
            with open(args.log_file, "w") as f:
                json.dump(history, f, indent=2)

    with open(args.log_file, "w") as f:
        json.dump(history, f, indent=2)

    print(f"\n[{args.optimizer.upper()} Seed {args.seed}] COMPLETE! 300M tokens processed in {wallclock:.1f}s.", flush=True)

if __name__ == "__main__":
    main()
