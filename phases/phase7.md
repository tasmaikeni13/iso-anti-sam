# Phase 7: Frozen 125M-Parameter, 1B-Token Confirmatory Pretraining on 8x MI300X

Work autonomously in the IsoAntiSAM repository and complete Phase 7. Read `phases/README.md` first and require a validated `PASS` handoff from Phase 6. This phase executes the frozen 125M confirmatory protocol; it does not alter hyperparameters or redesign the optimizer.

## 1. Objective
Execute the preregistered confirmatory pretraining of an approximately **125M-parameter** causal Transformer for exactly **1,000,000,000** non-padding FineWeb-Edu tokens across the **8x AMD Instinct MI300X** cluster. Compare IsoAntiSAM, standard AdamW, and standard SAM across three independent random seeds (`[42, 43, 44]`) under strictly equal compute budgets and identical data streams.

## 2. Required Work
1. **Preflight Protocol Integrity Check**:
   - Verify that the git commit, environment, hardware cluster, and data stream match the SHA256 checksum in `experiments/protocols/phase7_125m_protocol.json`.
   - Ensure all 8 MI300X GPUs are detected and idle with full 192–205 GB VRAM headroom.
   - Refuse execution if any post-hoc protocol modification is detected.
2. **Distributed Training Execution**:
   - Launch training via `torchrun --nproc_per_node=8 benchmarks/train_transformer.py`.
   - Train each optimizer (IsoAntiSAM, AdamW, SAM) across seeds `[42, 43, 44]`.
   - Ensure identical data presentation: for any given seed, all optimizers encounter exactly the same token sequence and micro-batch ordering.
   - Enforce zero post-hoc tuning: do not modify learning rate, batch size, or perturbation radius during or after inspection.
3. **Telemetry & Online Curvature Tracking**:
   - Record validation loss every 10M tokens on the held-out FineWeb-Edu validation slice.
   - Track online throughput (tokens/sec), MFU, step latency, peak VRAM per rank, and gradient coherence ratio $\cos(g_1, g_2)$.
   - Measure online Hessian spectral sharpness $\lambda_{\max}(H)$ via power iteration every 100M tokens.
4. **Resumable Checkpointing & Fault Tolerance**:
   - Save atomic model and optimizer checkpoints every 250M tokens.
   - In case of transient hardware or node failure, resume deterministically from the latest atomic checkpoint without altering the token cursor.
5. **Post-Run Verification**:
   - Confirm that each completed run processed exactly 1,000,000,000 non-padding tokens.
   - Verify that validation tokens remained strictly isolated from training.
6. **Specialized Skill Consultation**:
   - For long-run cluster telemetry, fault-tolerant checkpointing, and execution monitoring: activate and consult `skills/experimental-research` (`references/simulation-and-measurement.md`, `references/research-record.md`).
   - For protocol adherence, random seed control, and learning curve tracking: activate and consult `skills/ml-research` (`references/experiment-protocol.md`).
   - *Protocol Rule*: Use and apply these skills internally whenever needed, but do NOT cite skill names, meta-instructions, or tool invocations in final reports or scientific outputs.

## 3. Gate Criteria
Phase 7 passes only if:
- All confirmatory runs across seeds `[42, 43, 44]` for IsoAntiSAM, AdamW, and SAM complete successfully or fail under preregistered rules.
- Each run processes exactly 1B FineWeb-Edu tokens on the 125M model across 8x MI300X GPUs.
- IsoAntiSAM achieves a final validation perplexity improvement of **$\ge 1.0$ PPL** over tuned AdamW under identical tokens and FLOPs.
- Zero loss spikes, zero NaNs, and zero inter-rank divergence occur throughout pretraining.
- Standard Phase 7 artifacts (`report.md`, `manifest.json`, `commands.log`, `phases/status/phase7.json`) are written.
