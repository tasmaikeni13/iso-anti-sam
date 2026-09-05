# Phase 9: Cross-Scale Generalization Analysis, Downstream Benchmarks & Final Paper

Work autonomously in the IsoAntiSAM repository and complete Phase 9. Read `phases/README.md` first and require a validated `PASS` handoff from Phase 8. This phase consolidates all empirical and theoretical findings into publication-ready research artifacts.

## 1. Objective
Execute downstream zero-shot evaluation of pretrained checkpoints, extract empirical neural scaling exponents across model scales (14M, 125M, 350M), conduct ablation diagnostics on the isochoric gauge and bilateral gating mechanisms, and compile the final publication-ready manuscript in LaTeX.

## 2. Required Work
1. **Downstream Zero-Shot Evaluation**:
   - Evaluate the final checkpoints of the 125M and 350M models using `lm-evaluation-harness` across standard reasoning and knowledge benchmarks:
     - ARC-Easy / ARC-Challenge
     - HellaSwag
     - PIQA
     - MMLU (5-shot)
     - Lambada
   - Compare zero-shot accuracy between IsoAntiSAM and AdamW checkpoints.
2. **Cross-Scale Empirical Scaling Analysis**:
   - Fit compute-optimal scaling laws: compute cross-entropy validation loss as a function of training FLOPs:
     $$L(C) = \left(\frac{C_c}{C}\right)^{\alpha} + L_\infty$$
   - Extract the scaling exponent $\alpha$ and verify whether IsoAntiSAM shifts the Pareto frontier outward.
3. **Rigorous Component Ablations**:
   - Conduct controlled ablation experiments on 1x MI300X:
     - **Ablation 1**: IsoAntiSAM with Bilateral Gating turned OFF ($\text{gate} = 1$, raw micro-batch erosion). Measure needle attraction and overfitting.
     - **Ablation 2**: IsoAntiSAM with Isochoric Gauge turned OFF (unprojected Anti-SAM). Measure Hessian eigenvalue $\lambda_{\max}$ explosion.
     - **Ablation 3**: Static $\rho$ vs Cosine Perturbation Schedule.
4. **Final Research Paper & Reproducibility Ledger**:
   - Update `paper/paper.tex` with all final empirical figures, tables, downstream zero-shot scores, and Lean 4 formal proof citations.
   - Compile PDF via `pdflatex` / `latexmk` and verify zero compilation warnings.
   - Compile comprehensive artifact manifest linking every claim to its exact seed, log file, and checkpoint hash.

## 3. Gate Criteria
Phase 9 passes only if:
- Downstream zero-shot evaluation is complete across all standard tasks without missing evaluations.
- IsoAntiSAM demonstrates statistically significant improvements in downstream accuracy and token efficiency over AdamW.
- All ablations confirm that both the isochoric gauge and bilateral gating are necessary components of the optimizer.
- `paper/paper.pdf` builds cleanly without LaTeX errors or missing citations.
- Complete reproducibility ledger and final Phase 9 artifacts (`report.md`, `manifest.json`, `commands.log`, `phases/status/phase9.json`) are committed.
