# Phase 8: Frozen 350M-Parameter, 3B-Token Flagship Pretraining on 8x MI300X

Work autonomously in the IsoAntiSAM repository and complete Phase 8. Read `phases/README.md` first and require a validated `PASS` handoff from Phase 7. This phase executes the large-scale flagship pretraining protocol.

## 1. Objective
Scale IsoAntiSAM to an approximately **350M-parameter** causal Transformer trained across **3,000,000,000** (3 Billion) tokens of FineWeb-Edu on the **8x AMD Instinct MI300X** cluster (full-node HBM3 capacity: 1.536 TB). Evaluate empirical scaling laws, token efficiency gains, and large-scale optimization stability relative to tuned AdamW.

## 2. Required Work
1. **Preflight Cluster Verification**:
   - Verify cluster health across all 8 MI300X GPUs via `rocm-smi` and RCCL all-reduce smoke tests.
   - Verify protocol checksum for `experiments/protocols/phase8_350m_protocol.json`.
   - Verify local SSD storage margin for multi-gigabyte atomic checkpoints and dataset token shards.
2. **Distributed Flagship Execution**:
   - Launch `torchrun --nproc_per_node=8 benchmarks/train_transformer.py` in BF16 mixed precision with FlashAttention.
   - Model architecture: 24 layers, hidden dimension 1024, 16 attention heads, intermediate dimension 4096, sequence length 2048 (~350M parameters).
   - Global batch size: 1.0M tokens (512 sequences $\times$ 2048 length).
   - Enforce identical token ordering and random seeds between IsoAntiSAM and baseline AdamW.
3. **Token Efficiency & Scaling Law Evaluation**:
   - Evaluate the Token Efficiency Ratio: measure the number of tokens required by IsoAntiSAM to reach AdamW's final validation loss.
   - Verify whether IsoAntiSAM achieves the target validation loss using **15–20% fewer tokens** than AdamW.
   - Track Hessian spectral sharpness $\lambda_{\max}(H)$ to demonstrate that IsoAntiSAM maintains bounded, non-degenerate curvature even in wide deep networks.
4. **Telemetry & Resumption**:
   - Log high-resolution metrics every 10M tokens (`metrics.jsonl`).
   - Save distributed atomic checkpoints every 500M tokens. Maintain automatic deterministic resumption.
5. **Specialized Skill Consultation**:
   - For empirical scaling law analysis, token efficiency evaluation, and learning dynamics: activate and consult `skills/ml-research` (`references/research-loop.md`).
   - For full-node cluster measurement, HBM3 memory optimization, and throughput telemetry: activate and consult `skills/experimental-research` (`references/simulation-and-measurement.md`).
   - *Protocol Rule*: Use and apply these skills internally whenever needed, but do NOT cite skill names, meta-instructions, or tool invocations in final reports or scientific outputs.

## 3. Gate Criteria
Phase 8 passes only if:
- Full 3B-token pretraining completes on 8x MI300X without hardware drops, memory leaks, or NaN events.
- Sustained cluster throughput exceeds **180,000 tokens/second** (> 38% MFU).
- IsoAntiSAM demonstrates superior token efficiency: reaches AdamW's target validation loss using at least **15% fewer tokens**.
- Final validation perplexity is strictly superior to AdamW on held-out FineWeb-Edu.
- Standard Phase 8 artifacts (`report.md`, `manifest.json`, `commands.log`, `phases/status/phase8.json`) are written.
