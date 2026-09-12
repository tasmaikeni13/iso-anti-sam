# Parallel Pretraining Benchmark: 125M Model on 300M Tokens (FineWeb-Edu)

This directory contains the parallel pretraining execution harness and live benchmark telemetry across **6 concurrent runs** on a single **AMD Instinct MI300X GPU (192 GB HBM3, 750W TDP)**.

---

## 1. Experimental Overview & Matrix

| Run ID | Optimizer | Seed | Learning Rate ($\eta$) | Perturbation Radius ($\rho_0$) | Schedule | Batch Size (Tokens/Step) | Total Budget |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `adamw_seed42` | **AdamW** (Fused ROCm) | 42 | $6 \times 10^{-4}$ | — | Cosine with 3% warmup | 65,536 tokens | 300,000,000 |
| `adamw_seed43` | **AdamW** (Fused ROCm) | 43 | $6 \times 10^{-4}$ | — | Cosine with 3% warmup | 65,536 tokens | 300,000,000 |
| `sam_seed42` | **Vanilla SAM** | 42 | $6 \times 10^{-4}$ | $0.02$ | Constant | 65,536 tokens | 300,000,000 |
| `sam_seed43` | **Vanilla SAM** | 43 | $6 \times 10^{-4}$ | $0.02$ | Constant | 65,536 tokens | 300,000,000 |
| `carve_seed42` | **CARVE** (Isochoric Bilateral) | 42 | $6 \times 10^{-4}$ | $0.02$ | Cosine decay | 65,536 tokens | 300,000,000 |
| `carve_seed43` | **CARVE** (Isochoric Bilateral) | 43 | $6 \times 10^{-4}$ | $0.02$ | Cosine decay | 65,536 tokens | 300,000,000 |

- **Total Tokens across all 6 runs**: **1.8 Billion tokens** ($6 \times 300\text{M}$).
- **Sequence Length ($T$)**: 2048 tokens.
- **Batch Specification**: Micro-batch size 8 $\times$ Gradient accumulation 4 = 32 sequences per step = **65,536 tokens/step** (4,578 optimizer steps total per run).

---

## 2. Hardware Saturation & Kernel Optimization on MI300X

To maximize hardware utilization (MFU) and avoid idle GPU cycles, all 6 runs are launched **concurrently** on `cuda:0` rather than sequentially:
1. **FlashAttention SDPA & Fused Operations**: Native PyTorch `F.scaled_dot_product_attention` utilizing ROCm C++ FlashAttention kernels.
2. **Fused AdamW**: `fused=True` utilizing ROCm C++ vector math kernels for parameter updates.
3. **BFloat16 Mixed Precision**: Full autocast on all forward and backward passes with zero NaN/Inf instability.
4. **Zero-Copy Memory-Mapped Data Shards**: Custom `MmapTokenDataset` with `num_workers=0` for zero IPC serialization overhead, accessing `/root/carve/data/fineweb_edu_300m/train_tokens.npy` (1.2 GB, 300M tokens) directly in host memory.

### Live Telemetry (ROCm-SMI Snapshot)
- **Active Power Draw**: **747.0 W** (Saturating 99.6% of the 750.0W hardware power limit).
- **GPU Engine Activity**: **100% busy**.
- **Allocated VRAM**: **164.8 GB** / 192.0 GB (85.8% capacity, optimal utilization without driver paging).
- **Junction Temperature**: **73.0°C** (Well within thermal operating envelope).

---

## 3. Live Throughput & Completion ETA

As measured across the steady-state execution of all 6 concurrent workers:

| Metric | Measured Value |
| :--- | :--- |
| **Aggregate MI300X Throughput** | **~91,640 tokens / second** |
| **AdamW Throughput (Per Worker)** | ~24,000 tokens / second |
| **SAM Throughput (Per Worker)** | ~11,280 tokens / second |
| **CARVE Throughput (Per Worker)** | ~10,600 tokens / second |
| **Estimated Wallclock Remaining** | ~7 hours 45 minutes to 7 hours 53 minutes |
| **Estimated Completion Time (IST)** | **~12:15 AM – 12:25 AM IST (September 13, 2026)** |

---

## 4. Live Monitoring Instructions

To inspect real-time progress, loss convergence, per-run tokens seen, and live IST ETA at any moment:

```bash
python3 /root/carve/temp/monitor_runs.py
```

To view raw unbuffered standard output logs for an individual run:

```bash
tail -f /root/carve/temp/logs/carve_seed42.out
tail -f /root/carve/temp/logs/adamw_seed42.out
tail -f /root/carve/temp/logs/sam_seed42.out
```

---

## 5. Artifacts and Directory Structure

```
temp/
├── README.md               # This documentation report
├── model.py                # 125M Causal Transformer with RoPE, RMSNorm, SwiGLU, SDPA
├── train_worker.py         # Worker script executing individual pretraining runs
├── launch_parallel.py      # Orchestrator & supervisor managing 6 concurrent workers
├── monitor_runs.py         # Real-time console monitor with IST calculations
├── parallel_pids.json      # Manifest tracking active process IDs
└── logs/
    ├── adamw_seed42_history.json
    ├── adamw_seed43_history.json
    ├── sam_seed42_history.json
    ├── sam_seed43_history.json
    ├── carve_seed42_history.json
    └── carve_seed43_history.json
```
