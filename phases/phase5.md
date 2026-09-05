# Phase 5: Small-Scale Falsification, Multi-Workload Screening & WikiText-103 Baselines

Work autonomously in the IsoAntiSAM repository and complete Phase 5. Read `phases/README.md` first and require a validated `PASS` handoff from Phase 4. This phase tests whether the morphological erosion advantage and caustic protection hold in real autoregressive language modeling workloads before large-scale pretraining.

## 1. Objective
Establish an empirical screening pipeline on real NLP benchmarks using a 6-layer causal Transformer on **WikiText-103**. Evaluate IsoAntiSAM against established baselines (AdamW, SGD with momentum, and Standard SAM) under identical token streams, compute budgets, and sequence length. Confirm that IsoAntiSAM avoids the caustic overfitting of Raw Anti-SAM and exhibits synchronous training and validation descent.

## 2. Required Work
1. **WikiText-103 Data Pipeline & Sharding**:
   - Download WikiText-103 and tokenize into deterministic binary memory-mapped shards (`train_tokens.npy`, `val_tokens.npy`) using a 10,000-token BPE / byte vocabulary.
   - Establish high-throughput PyTorch dataset and zero-copy dataloaders with fixed sequence length $L = 256$.
2. **Model Architecture & Acceleration**:
   - Construct a 6-layer causal Transformer (~14.59M parameters, hidden dimension 384, 6 attention heads).
   - Integrate native PyTorch FlashAttention via Scaled Dot-Product Attention (`F.scaled_dot_product_attention(is_causal=True)`), dispatching directly to AMD ROCm CK/AOTriton kernels on `gfx942`.
3. **Equal-Budget Baseline Screening**:
   - Execute controlled, compute-matched benchmark runs across:
     - **AdamW Baseline** ($lr = 5 \times 10^{-4}$, weight decay 0.01).
     - **SGD Baseline** ($lr = 0.05$, momentum 0.9).
     - **Standard SAM** ($\rho = 0.05$, AdamW base).
     - **Raw Anti-SAM** ($\rho = 0.05$, AdamW base).
     - **IsoAntiSAM (Ours)** ($\rho = 0.05$, bilateral coherence gating, AdamW base).
   - Enforce identical batch size ($B = 64$), identical cosine learning rate schedule with warmup, and identical random seed (`seed=42`).
4. **Generalization & Overfitting Diagnostics**:
   - Measure validation cross-entropy loss, perplexity (PPL), and epoch training duration.
   - Track the generalization divergence gap: $\Delta(t) = |L_{\text{val}}(t) - L_{\text{train}}(t)|$.
   - Verify that IsoAntiSAM avoids the rapid validation divergence characteristic of Raw Anti-SAM.
5. **Specialized Skill Consultation**:
   - For language model benchmark design and equal-budget baseline protocols: activate and consult `skills/ml-research` (`references/research-loop.md`, `references/method-search.md`).
   - For empirical data pipelining, token shard integrity, and memory accounting: activate and consult `skills/experimental-research` (`references/study-design.md`).
   - *Protocol Rule*: Use and apply these skills internally whenever needed, but do NOT cite skill names, meta-instructions, or tool invocations in final reports or scientific outputs.

## 3. Gate Criteria
Phase 5 passes only if:
- WikiText-103 token shards are fully generated and verified with deterministic token counts.
- 6-layer causal Transformer trains with **zero OOM errors** and **zero NaNs** on 1x AMD Instinct MI300X.
- All baseline optimizer histories are logged to structured JSON files under `logs/wikitext/`.
- Training loss and validation loss descend synchronously under IsoAntiSAM without catastrophic divergence.
- Standard Phase 5 artifacts (`report.md`, `manifest.json`, `commands.log`, `phases/status/phase5.json`) are written.
