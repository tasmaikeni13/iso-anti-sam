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
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

sys.path.insert(0, "/root/carve")
sys.path.insert(0, "/root/carve/src")
from carve import Carve, SAM

from hyperparameters.model_transformer import create_model_125m, create_model_350m
from hyperparameters.fineweb_loader import prepare_fineweb_edu_shard, get_dataloaders

def get_cosine_schedule_with_warmup(step, warmup_steps, max_steps, base_lr, min_lr_ratio=0.1):
    if step < warmup_steps:
        return base_lr * float(step) / float(max(1, warmup_steps))
    progress = float(step - warmup_steps) / float(max(1, max_steps - warmup_steps))
    return base_lr * (min_lr_ratio + 0.5 * (1.0 - min_lr_ratio) * (1.0 + math.cos(math.pi * progress)))

def get_rho_schedule(step, max_steps, rho_0):
    progress = min(1.0, float(step) / float(max(1, max_steps)))
    return rho_0 * 0.5 * (1.0 + math.cos(math.pi * progress))

def compute_hessian_max_eigenvalue(model, x, y, n_iters=3, r=1e-4):
    device = x.device
    params = [p for p in model.parameters() if p.requires_grad]
    v = [torch.randn_like(p, device=device) for p in params]
    v_norm = torch.sqrt(sum((vi**2).sum() for vi in v))
    v = [vi / (v_norm + 1e-12) for vi in v]

    model.zero_grad()
    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        out = model(x)
        loss = F.cross_entropy(out.view(-1, out.size(-1)), y.view(-1))
    loss.backward(create_graph=True)
    g_base = [p.grad.clone() for p in params]

    for _ in range(n_iters):
        with torch.no_grad():
            for p, vi in zip(params, v):
                p.add_(vi * r)
        model.zero_grad()
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            out_p = model(x)
            loss_p = F.cross_entropy(out_p.view(-1, out_p.size(-1)), y.view(-1))
        loss_p.backward()
        g_pert = [p.grad.clone() for p in params]
        model.zero_grad()

        with torch.no_grad():
            for p, vi in zip(params, v):
                p.sub_(vi * r)

        Hv = [(gp - gb) / r for gp, gb in zip(g_pert, g_base)]
        lambda_max = sum((vi * hvi).sum().item() for vi, hvi in zip(v, Hv))
        Hv_norm = torch.sqrt(sum((hvi**2).sum() for hvi in Hv))
        v = [hvi / (Hv_norm + 1e-12) for hvi in Hv]

    model.zero_grad()
    return float(max(0.0, lambda_max))

def parse_args():
    parser = argparse.ArgumentParser(description="Distributed FineWeb-Edu LLM Training (AdamW, SAM, CARVE)")
    parser.add_argument("--scale", type=str, default="125m", choices=["125m", "350m"])
    parser.add_argument("--optimizer", type=str, default="carve", choices=["adamw", "sam", "carve"])
    parser.add_argument("--lr", type=float, default=6e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--rho", type=float, default=0.02)
    parser.add_argument("--rho_schedule", type=str, default="cosine", choices=["cosine", "constant"])
    parser.add_argument("--coherence_floor", type=float, default=0.0)
    parser.add_argument("--seq_len", type=int, default=2048)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum_steps", type=int, default=16)
    parser.add_argument("--max_steps", type=int, default=100)
    parser.add_argument("--warmup_steps", type=int, default=10)
    parser.add_argument("--val_interval", type=int, default=20)
    parser.add_argument("--val_steps", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data_dir", type=str, default="/root/carve/data/fineweb_edu")
    parser.add_argument("--log_dir", type=str, default="/root/carve/logs/hyperparameters")
    parser.add_argument("--exp_name", type=str, default=None)
    return parser.parse_args()

def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.log_dir, exist_ok=True)

    is_distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    if is_distributed:
        dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            torch.cuda.set_device(device)
    else:
        rank = 0
        world_size = 1
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if rank == 0:
        print(f"=== Starting Training: {args.optimizer.upper()} | Model: {args.scale.upper()} ===")
        print(f"Device: {device} | Distributed: {is_distributed} (World size: {world_size})")

    # Data preparation
    if rank == 0:
        train_path, val_path = prepare_fineweb_edu_shard(target_tokens=50_000_000, save_dir=args.data_dir, seed=args.seed)
    if is_distributed:
        dist.barrier()
        train_path = os.path.join(args.data_dir, "train_tokens.npy")
        val_path = os.path.join(args.data_dir, "val_tokens.npy")

    train_loader, val_loader = get_dataloaders(
        train_path, val_path, seq_len=args.seq_len,
        batch_size=args.batch_size, num_workers=2
    )
    train_iter = iter(train_loader)

    # Model construction
    if args.scale == "125m":
        model = create_model_125m(vocab_size=50257, max_seq_len=args.seq_len).to(device)
    else:
        model = create_model_350m(vocab_size=50257, max_seq_len=args.seq_len).to(device)

    total_params = model.get_num_params()
    if rank == 0:
        print(f"Total Model Parameters: {total_params:,}")

    if is_distributed:
        model = DDP(model, device_ids=[local_rank] if torch.cuda.is_available() else None)

    raw_model = model.module if is_distributed else model

    # Optimizer initialization
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

    history = {
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
        "lambda_max": []
    }

    tokens_per_step = args.batch_size * args.seq_len * args.grad_accum_steps * world_size
    total_tokens_seen = 0

    exp_prefix = args.exp_name if args.exp_name else f"{args.scale}_{args.optimizer}"
    log_file = os.path.join(args.log_dir, f"{exp_prefix}_history.json")

    start_time = time.time()

    for step in range(1, args.max_steps + 1):
        step_start = time.time()
        curr_lr = get_cosine_schedule_with_warmup(step, args.warmup_steps, args.max_steps, args.lr)
        for pg in optimizer.param_groups:
            pg["lr"] = curr_lr

        if args.rho_schedule == "cosine":
            curr_rho = get_rho_schedule(step, args.max_steps, args.rho)
        else:
            curr_rho = args.rho

        model.train()
        accum_loss = 0.0
        coherence_val = 0.0

        if args.optimizer == "carve":
            # CARVE Two-phase bilateral stepping with micro-batches
            optimizer.zero_grad(set_to_none=True)
            try:
                x_full, y_full = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                x_full, y_full = next(train_iter)

            x_full, y_full = x_full.to(device, non_blocking=True), y_full.to(device, non_blocking=True)
            half_b = max(1, x_full.size(0) // 2)

            # Micro-batch 1
            with torch.amp.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu", dtype=torch.bfloat16):
                out1 = model(x_full[:half_b])
                loss1 = F.cross_entropy(out1.view(-1, 50257), y_full[:half_b].view(-1))
            loss1.backward()
            g1 = [p.grad.clone() if p.grad is not None else None for p in model.parameters()]

            # Micro-batch 2
            model.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu", dtype=torch.bfloat16):
                out2 = model(x_full[half_b:])
                loss2 = F.cross_entropy(out2.view(-1, 50257), y_full[half_b:].view(-1))
            loss2.backward()
            g2 = [p.grad.clone() if p.grad is not None else None for p in model.parameters()]

            # Compute bilateral perturbation
            model.zero_grad(set_to_none=True)
            coherence_val = optimizer.compute_bilateral_perturbation(g1, g2, rho=curr_rho)

            # Outer evaluation on perturbed weights
            with torch.amp.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu", dtype=torch.bfloat16):
                out_full = model(x_full)
                loss_full = F.cross_entropy(out_full.view(-1, 50257), y_full.view(-1))
            loss_full.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step_with_bilateral(zero_grad=True)
            accum_loss = loss_full.item()

        elif args.optimizer == "sam":
            optimizer.zero_grad(set_to_none=True)
            try:
                x, y = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                x, y = next(train_iter)
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

            with torch.amp.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu", dtype=torch.bfloat16):
                out1 = model(x)
                loss1 = F.cross_entropy(out1.view(-1, 50257), y.view(-1))
            loss1.backward()
            optimizer.first_step(zero_grad=True)

            with torch.amp.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu", dtype=torch.bfloat16):
                out2 = model(x)
                loss2 = F.cross_entropy(out2.view(-1, 50257), y.view(-1))
            loss2.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.second_step(zero_grad=True)
            accum_loss = loss2.item()

        elif args.optimizer == "adamw":
            optimizer.zero_grad(set_to_none=True)
            for _ in range(args.grad_accum_steps):
                try:
                    x, y = next(train_iter)
                except StopIteration:
                    train_iter = iter(train_loader)
                    x, y = next(train_iter)
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

                with torch.amp.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu", dtype=torch.bfloat16):
                    out = model(x)
                    loss = F.cross_entropy(out.view(-1, 50257), y.view(-1)) / args.grad_accum_steps
                loss.backward()
                accum_loss += loss.item()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        step_dur = time.time() - step_start
        total_tokens_seen += tokens_per_step
        throughput = tokens_per_step / max(1e-4, step_dur)

        # Validation step
        val_loss = None
        val_ppl = None
        lambda_max = 0.0

        if step % args.val_interval == 0 or step == args.max_steps:
            model.eval()
            val_loss_tot = 0.0
            val_count = 0
            with torch.no_grad():
                for v_step, (vx, vy) in enumerate(val_loader):
                    if v_step >= args.val_steps:
                        break
                    vx, vy = vx.to(device, non_blocking=True), vy.to(device, non_blocking=True)
                    with torch.amp.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu", dtype=torch.bfloat16):
                        vout = model(vx)
                        vloss = F.cross_entropy(vout.view(-1, 50257), vy.view(-1))
                    val_loss_tot += vloss.item()
                    val_count += 1
            val_loss = val_loss_tot / max(1, val_count)
            val_ppl = math.exp(min(val_loss, 20.0))

            # Compute Hessian spectral sharpness on rank 0
            if rank == 0:
                sx, sy = next(iter(val_loader))
                sx, sy = sx[:2].to(device), sy[:2].to(device)
                lambda_max = compute_hessian_max_eigenvalue(raw_model, sx, sy, n_iters=2)

            if rank == 0:
                print(f"Step {step:4d}/{args.max_steps} | Train: {accum_loss:.4f} | Val: {val_loss:.4f} | PPL: {val_ppl:6.2f} | λ_max: {lambda_max:5.2f} | Cos(g1,g2): {coherence_val:+.3f} | {throughput:,.0f} tok/s")

        if rank == 0:
            history["step"].append(step)
            history["tokens_seen"].append(total_tokens_seen)
            history["train_loss"].append(accum_loss)
            history["val_loss"].append(val_loss if val_loss is not None else (history["val_loss"][-1] if history["val_loss"] else accum_loss))
            history["val_ppl"].append(val_ppl if val_ppl is not None else (history["val_ppl"][-1] if history["val_ppl"] else math.exp(min(accum_loss, 20.0))))
            history["learning_rate"].append(curr_lr)
            history["rho"].append(curr_rho)
            history["coherence_cos"].append(coherence_val)
            history["step_time_s"].append(step_dur)
            history["throughput_tokens_per_s"].append(throughput)
            history["lambda_max"].append(lambda_max)

    if rank == 0:
        with open(log_file, "w") as f:
            json.dump(history, f, indent=2)
        print(f"\nTraining completed for {exp_prefix}. Saved to {log_file}")

    if is_distributed:
        dist.destroy_process_group()

if __name__ == "__main__":
    main()
