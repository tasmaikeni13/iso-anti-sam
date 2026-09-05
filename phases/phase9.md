# Phase 9: 350M Parameter Model Pretraining on 3B Tokens FineWeb-Edu on 8x AMD MI300X

## 1. Objective
Scale the IsoAntiSAM optimizer to a 350 Million parameter causal Transformer trained across 3 Billion tokens of FineWeb-Edu on 8x AMD Instinct MI300X GPUs (full node scale).
Evaluate empirical scaling laws, wall-clock efficiency, and generalization transfer.

## 2. Model & Training Architecture
- Parameters: 350 Million (Layers: 24, Hidden Dim: 1024, Heads: 16, Intermediate Dim: 4096, Max Seq Len: 2048).
- Tokens: 3,000,000,000 tokens (3B tokens) from FineWeb-Edu.
- Hardware: 8x AMD Instinct MI300X (total 1536 GB HBM3 memory, interconnected via high-bandwidth infinity fabric).
- Precision: BF16 mixed precision with native ROCm FlashAttention.
- Target Metric: Validation cross-entropy loss, token throughput, downstream zero-shot accuracy.

## 3. Execution Commands
```bash
torchrun --nproc_per_node=8 /root/iso-anti-sam/benchmarks/train_transformer.py \
  --model_size 350M \
  --tokens 3000000000 \
  --optimizer iso_anti_sam \
  --rho 0.05 \
  --lr 3e-4 \
  --batch_size 16 \
  --grad_accum 16 \
  --bf16 \
  --data_path /root/iso-anti-sam/data/fineweb_3B \
  --output_dir /root/iso-anti-sam/checkpoints/350M_iso_anti_sam
```

## 4. Expected Outputs & Success Criteria
1. Throughput $> 180,000$ tokens/second sustained across 8x MI300X.
2. Complete 3B token run without GPU memory fragmentation or hardware drops.
3. IsoAntiSAM demonstrates superior token efficiency: achieves the target validation loss of AdamW using 20–25% fewer tokens.
4. Validation loss tracks training loss descent with zero sharp-needle overfitting.

## 5. Self-Correcting Autonomous Fallback Loop
If scaling breakdown or multi-GPU synchronization stall occurs:
1. **Activate `experimental-research` skill** (`/root/skills_repo/experimental-research/SKILL.md`) and consult `references/study-design.md`.
2. Inspect gradient all-reduce synchronization across ROCm RCCL streams: ensure micro-batch gradients $g_1, g_2$ are all-reduced independently or fused into a single interleaved tensor to halve communication overhead.
3. Check gradient clipping and BF16 numerical dynamic range.
4. If loss spike occurs, apply autonomous rollback to preceding checkpoint, adjust $\rho$, and resume.
