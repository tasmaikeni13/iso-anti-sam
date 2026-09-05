# Phase 8: 125M Parameter Model Pretraining on 1B Tokens FineWeb-Edu (1x/8x AMD MI300X)

## 1. Objective
Train a 125M parameter modern causal decoder Transformer on 1 Billion tokens of FineWeb-Edu to evaluate the scaling behavior and generalization dynamics of IsoAntiSAM against AdamW and SAM.

## 2. Model & Training Architecture
- Parameters: 125 Million (Layers: 12, Hidden Dim: 768, Heads: 12, Intermediate Dim: 3072, Max Seq Len: 2048).
- Modern Architecture: RoPE (Rotary Position Embeddings), RMSNorm, SwiGLU activations, FlashAttention / SDPA.
- Tokens: 1,000,000,000 tokens (1B tokens) from FineWeb-Edu.
- Batch Size: Global batch size 0.5M tokens (256 sequences of length 2048).
- Hardware: AMD Instinct MI300X (1x or distributed across 8x MI300X).
- Optimizers Compared: AdamW (baseline), SAM ($\rho=0.05$), IsoAntiSAM ($\rho=0.05$).

## 3. Execution Commands
```bash
# Launch 125M run on 8x MI300X using torchrun
torchrun --nproc_per_node=8 /root/iso-anti-sam/benchmarks/train_transformer.py \
  --model_size 125M \
  --tokens 1000000000 \
  --optimizer iso_anti_sam \
  --rho 0.05 \
  --lr 6e-4 \
  --batch_size 32 \
  --grad_accum 8 \
  --data_path /root/iso-anti-sam/data/fineweb_1B \
  --output_dir /root/iso-anti-sam/checkpoints/125M_iso_anti_sam
```

## 4. Expected Outputs & Success Criteria
1. Full 1B token pretraining completed in targeted wall-clock time.
2. Validation loss curve logged every 10M tokens.
3. IsoAntiSAM validation loss drops strictly faster than AdamW throughout the pretraining trajectory.
4. Final validation perplexity improvement $\ge 1.0$ PPL over AdamW under identical FLOPs and tokens.
5. Zero divergence or loss spikes.

## 5. Self-Correcting Autonomous Fallback Loop
If validation loss diverges or fails to improve:
1. **Activate `ml-research` skill** (`/root/skills_repo/ml-research/SKILL.md`) and consult `references/research-loop.md`.
2. Inspect the gradient coherence ratio $\cos(g_1, g_2)$ over training steps:
   - If coherence drops below 0.1 at late steps, decay perturbation radius $\rho_t \propto \frac{1}{\sqrt{t}}$ to prevent late-stage gradient diffusion.
   - If learning rate warmup is too aggressive, extend warmup from 1,000 to 2,500 steps.
3. If an invariant is violated, re-derive the stability criteria using `theory-research`, update Lean 4 proofs, and resume training from the latest clean checkpoint.
