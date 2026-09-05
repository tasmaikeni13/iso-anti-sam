# Phase 2: Numerical Verification, Landscape Geometry, and Transverse Curvature Diagnostics

## 1. Objective
Perform empirical and numerical verification of the mathematical derivations across controlled multi-dimensional loss landscapes:
1. Measure the first-order training loss drop of Anti-SAM vs SAM vs SGD.
2. Measure the dilation of sharp-needle catchment basins and quantify trapping probability.
3. Measure the empirical divergence $\operatorname{div}(E)$ across parameter dimensions $d \in [10, 1000]$ to verify the Caustic Collapse Theorem.
4. Verify that Bilateral Coherent Anti-SAM successfully eliminates needle trapping and synchronizes validation descent.

## 2. Execution Commands
```bash
python3 /root/iso-anti-sam/analysis/numerical_analysis.py
```

## 3. Expected Outputs & Success Criteria
1. Execution completes with code 0.
2. Plots generated in `/root/iso-anti-sam/analysis/figures/`:
   - `simulation_landscape.png`
   - `divergence_scaling.png`
3. Quantitative confirmation:
   - Anti-SAM exhibits rapid early training drop, but high validation loss when sample needles exist.
   - IsoAntiSAM validation loss is strictly lower than standard Anti-SAM ($\Delta L_{\text{val}} \le 0.8 \times L_{\text{anti}}$).
   - Divergence scaling confirms $\operatorname{div}(E_{\text{anti}}) \propto -d$, whereas $\operatorname{div}(E_{\text{iso}}) = 0$.

## 4. Self-Correcting Autonomous Fallback Loop
If numerical results show unexpected behavior (e.g. divergence does not scale negatively or IsoAntiSAM fails to generalize):
1. **Activate `experimental-research` skill** (`/root/skills_repo/experimental-research/SKILL.md`) and consult `references/simulation-and-measurement.md`.
2. Check finite-difference step size $\epsilon$ for floating-point underflow/catastrophic cancellation.
3. Check random seeds and condition numbers of the Hessian matrix.
4. If the bilateral coherence gate $\mathcal{C}$ collapses prematurely due to high noise variance, consult `mechanism-transfer` and apply exponential moving average (EMA) smoothing to the cross-batch inner product.
5. Re-run `numerical_analysis.py` until all quantitative assertions pass.
