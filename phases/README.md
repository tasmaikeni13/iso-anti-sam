# Autonomous Research Protocol & State Machine: IsoAntiSAM

These nine sequential phase files are copy-ready prompts for fresh autonomous agent and researcher sessions. They govern the end-to-end scientific lifecycle of **IsoAntiSAM** (Isochoric and Coherent Anti-Sharpness-Aware Minimization).

A downstream phase may begin execution **only** when the preceding phase has committed a validated `PASS` handoff in `phases/status/`. The prompts authorize in-scope code development, formal proof construction, GPU benchmarking, non-force git commits, and pushes to this repository; they do not authorize destructive administrative actions, disclosure of secrets, or submission of manuscripts.

| Phase | Purpose | Expensive GPU Work |
|---|---|---|
| **1** | Adversarial novelty, morphological erosion theory, and Lean 4 formal proof audit | No |
| **2** | Landscape geometry, caustic collapse dynamics, and isochoric gauge divergence diagnostics | Small diagnostics only |
| **3** | PyTorch optimizer architecture, unit testing, and algorithmic invariants | Smoke / unit tests |
| **4** | Native AMD ROCm/HIP C++ fused coherent erosion kernel development & MI300X profiling | Kernel benchmarks |
| **5** | Small-scale falsification, WikiText-103 baseline screening, and empirical collapse verification | 1x MI300X sweeps |
| **6** | Scaling pilot, 8x MI300X multi-GPU distributed orchestration (RCCL DDP), and dual preregistration | Multi-GPU pilots |
| **7** | Frozen 125M-parameter, 1B-token confirmatory pretraining on FineWeb-Edu across 8x MI300X | Yes (1B tokens) |
| **8** | Frozen 350M-parameter, 3B-token flagship pretraining on FineWeb-Edu across 8x MI300X | Yes (3B tokens) |
| **9** | Cross-scale generalization analysis, downstream benchmarks, and publishable paper | Rechecks only |

---

## 1. Shared State Machine

Every phase must inspect the active repository state, preceding phase reports, the current machine architecture, and GPU memory before executing actions. Never assume environment packages, disk margins, or GPU availability.

Each phase writes four standard artifacts upon conclusion:
1. `artifacts/phaseN/report.md`: Technical decisions, quantitative evidence, failure modes, and the exact gate result.
2. `artifacts/phaseN/manifest.json`: Commit hash, hardware environment, exact CLI commands, random seeds, input/output paths, and SHA256 checksums of decisive artifacts.
3. `artifacts/phaseN/commands.log`: Shell commands sufficient to reproduce the phase from scratch (secrets redacted).
4. `phases/status/phaseN.json`: Machine-readable record containing `gate` (`PASS`, `REVISE`, `FAIL_CORE`, or `BLOCKED`), `phaseN_handoff_commit`, UTC timestamp, run ID, and certified rationale.

### Storage & Git Hygiene
- Heavy model checkpoints (`*.pt`), binary token shards (`*.npy`, `*.bin`), and raw datasets (`data/`) **must remain strictly outside Git** via `.gitignore`.
- Small structured logs (`metrics.jsonl`, `*_history.json`), figures, configuration JSONs, and audit reports must be committed to Git.
- Preserve all attempted runs (both passing and failing). Never overwrite or hide negative results.
- **Strict Prohibition**: Under no circumstances commit or push `prompt.md`.

---

## 2. Failure Routing & Classification

When a gate condition fails, classify the root cause into one of the following seven formal categories:

1. **Implementation Failure**: Defect in PyTorch/HIP code, memory allocation, or tensor indexing. *Action*: Fix locally in the current phase and rerun the smallest decisive regression test.
2. **Infrastructure Failure**: Driver timeout, ROCm/RCCL initialization error, or file I/O bottleneck. *Action*: Repair environment configuration, verify hardware visibility, and retry.
3. **Resource Failure**: Out-of-memory (OOM) error or insufficient disk space. *Action*: Adjust gradient accumulation, sequence packing, or checkpoint retention. Do not alter algorithmic definition.
4. **Experimental-Design Failure**: Unequal tuning budgets, hyperparameter mismatch, or data order leakage between compared optimizers. *Action*: Equalize evaluation budgets and rerun.
5. **Mathematical / Theoretical Failure**: Violation of the isochoric gauge condition ($\operatorname{div}_{T^\perp}(E) \neq 0$), failure of sample noise cancellation in bilateral inner products, or unexpected caustic singularity. *Action*: Trigger the **Theory-Repair Loop**. Reopen Phases 1 and 2, re-derive the mathematical formulation, update Lean 4 proofs, and invalidate affected downstream phases.
6. **Novelty / Collision Failure**: Discovery of identical prior art implementing bilateral cross-batch erosion or transverse isochoric perturbation fields. *Action*: Mark `FAIL_CORE`. Update literature matrix and derive a distinct non-colliding mathematical primitive.
7. **Empirical Failure**: Underperformance against AdamW or SAM under equal tuning budgets. *Action*: Diagnose gradient coherence ratios and perturbation schedules. If the mechanism itself is falsified, enter theory repair; otherwise, report the result with scientific fidelity.

---

## 3. Scientific and Operational Guardrails

### No "Cheating" Rescues
The core thesis of IsoAntiSAM is that **morphological erosion accelerated descent can be stabilized against sharp-needle caustics exclusively through transverse isochoric projection and bilateral coherence gating**.
- **Forbidden Rescues**: Never rescue underperforming runs by adding arbitrary momentum, ad-hoc gradient clipping heuristics, matrix whitening, polar decomposition steps, or fallback mixtures with AdamW/SGD.
- Any change to the perturbation field $e(w)$ or update equation must be derived from first principles, machine-checked in Lean 4, and audited for novelty.

### AMD Instinct MI300X Cluster Operations
- **Preflight Inspection**: Always run inventory diagnostics (`rocm-smi`, `hipcc --version`, PyTorch ROCm device count) before launching training.
- **Native ROCm Compilations**: Target `gfx942` using official ROCm 6.3 `/opt/rocm/bin/hipcc -O3`.
- **Attention Acceleration**: Use PyTorch Scaled Dot-Product Attention (`F.scaled_dot_product_attention(is_causal=True)`), which natively dispatches to ROCm CK / AOTriton FlashAttention kernels on MI300X.
- **Multi-GPU Orchestration**: Use PyTorch DistributedDataParallel (`torchrun --nproc_per_node=8`) with RCCL backend. Micro-batch gradients $g_1, g_2$ must be all-reduced across ranks before the isochoric perturbation is evaluated.
- **Numerical Precision**: Compute all scalar reductions (vector norms, cross-batch inner products, coherence gates) in FP32 or higher to prevent underflow. Execute model parameters in BF16 or FP32.

---

## 4. Execution Guidance

To run a phase, provide the entire phase prompt (`phases/phaseN.md`) to a fresh agent session. Ensure all preceding phase status files in `phases/status/` record `PASS`. Phases 7 and 8 are strictly isolated execution phases to allow multi-hour GPU training without design drift.
