# Phase 6: Scaling Pilot, 8x MI300X Multi-GPU Distributed Orchestration & Dual Preregistration

Work autonomously in the Carve repository and complete Phase 6. Read `phases/README.md` first and require a validated `PASS` handoff from Phase 5. This phase establishes multi-GPU distributed orchestration and freezes the confirmatory pretraining protocols before launching long GPU runs.

## 1. Objective
Establish multi-GPU distributed scaling across the **8x AMD Instinct MI300X** cluster using PyTorch DistributedDataParallel (`torchrun --nproc_per_node=8`) with AMD RCCL. Resolve perturbation schedule calibration for language models, benchmark multi-GPU throughput and MFU scaling, and preregister immutable protocols for training:
1. An approximately **125M-parameter** decoder-only Transformer on exactly **1,000,000,000** FineWeb-Edu tokens.
2. An approximately **350M-parameter** decoder-only Transformer on exactly **3,000,000,000** FineWeb-Edu tokens.

## 2. Required Work
1. **Perturbation Schedule Calibration (Small-Model Fallback Loop)**:
   - On the 14M Transformer, resolve late-stage gradient jitter by evaluating a cosine perturbation decay schedule:
     $$\rho_t = \rho_0 \cdot \frac{1}{2}\left(1 + \cos\left(\frac{\pi t}{T}\right)\right)$$
   - Calibrate base perturbation radius $\rho_0 \in [0.005, 0.01, 0.02, 0.05]$.
   - Confirm that with scheduled erosion decay, Carve outperforms standard AdamW in final validation perplexity by $\ge 1.0$ PPL point on WikiText-103.
2. **Multi-GPU Distributed Orchestration on 8x MI300X**:
   - Implement multi-GPU DDP support with AMD ROCm RCCL communication (`torch.distributed.init_process_group('nccl')`).
   - Implement distributed bilateral all-reduce: ensure micro-batch gradients $g_1, g_2$ are all-reduced across all 8 ranks before evaluating the coherence gate and isochoric perturbation.
   - Verify bitwise numerical agreement across ranks: all ranks must observe identical all-reduced norms and apply identical perturbation steps (zero inter-rank drift).
   - Benchmark throughput (tokens/sec) and Model FLOPs Utilization (MFU) across 1, 2, 4, and 8 MI300X GPUs.
3. **Freeze 125M / 1B Architecture & Data Protocol**:
   - Model parameters: 125M within $\pm 2\%$ (12 layers, hidden dimension 768, 12 heads, max sequence length 2048).
   - Modern architecture components: RoPE (Rotary Position Embeddings), RMSNorm, SwiGLU activations, FlashAttention SDPA.
   - Dataset: `HuggingFaceFW/fineweb-edu` (sample-10BT slice), tokenized into 2048-token sequence chunks.
   - Training budget: Exactly 1,000,000,000 non-padding tokens. Global batch size: 0.5M tokens (256 sequences $\times$ 2048 length).
   - Confirmatory seeds: `[42, 43, 44]`.
   - Write immutable protocol file to `experiments/protocols/phase7_125m_protocol.json`.
4. **Freeze 350M / 3B Architecture & Data Protocol**:
   - Model parameters: 350M within $\pm 2\%$ (24 layers, hidden dimension 1024, 16 heads, sequence length 2048).
   - Training budget: Exactly 3,000,000,000 tokens. Global batch size: 1.0M tokens.
   - Confirmatory seeds: `[42, 43, 44]`.
   - Write immutable protocol file to `experiments/protocols/phase8_350m_protocol.json`.
5. **Preregistration & Checksums**:
   - Compute SHA256 checksums of both protocol files.
   - Verify local disk space margin for FineWeb-Edu token shards and atomic checkpoints.
6. **Specialized Skill Consultation**:
   - For perturbation schedule design, fallback loops, and learning rate/radius co-adaptation: activate and consult `skills/ml-research` (`references/research-loop.md`, `references/method-search.md`).
   - For multi-GPU DDP scaling, MFU measurement, and throughput profiling: activate and consult `skills/experimental-research` (`references/simulation-and-measurement.md`).
   - For geometric parameterization and cross-family optimization transfer: activate and consult `skills/mechanism-transfer` (`references/mechanism-families.md`).
   - *Protocol Rule*: Use and apply these skills internally whenever needed, but do NOT cite skill names, meta-instructions, or tool invocations in final reports or scientific outputs.

## 3. Gate Criteria
Phase 6 passes only if:
- Scheduled perturbation decay is verified, resolving the small-model perplexity gap over AdamW.
- Multi-GPU DDP orchestration across 8x AMD MI300X is verified with zero inter-rank weight drift.
- Scaling efficiency across 8x MI300X exceeds **85% linear scaling**.
- Immutable protocol JSON files (`phase7_125m_protocol.json` and `phase8_350m_protocol.json`) are committed with verified SHA256 hashes.
- Standard Phase 6 artifacts (`report.md`, `manifest.json`, `commands.log`, `phases/status/phase6.json`) are written.
