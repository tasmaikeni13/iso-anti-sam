# Phase 2: Landscape Geometry, Caustic Collapse Dynamics & Transverse Divergence Diagnostics

Work autonomously in the Carve repository and complete Phase 2. Read `phases/README.md` first and require a validated `PASS` handoff from Phase 1. This phase is empirical and numerical; GPU compute is restricted to small diagnostic scripts.

## 1. Objective
Empirically falsify or substantiate the theoretical predictions from Phase 1 regarding caustic collapse and sharp-needle entrapment. Construct high-dimensional synthetic non-convex loss surfaces containing both wide generalizable basins and narrow, spurious sharp needles. Measure trajectory dynamics, basin volume dilation, and transverse divergence scaling to prove that Carve escapes the caustic catastrophe that traps raw Anti-SAM.

## 2. Required Work
1. **Synthetic Multi-Basin Test Landscape**:
   - Construct a parameterized high-dimensional loss surface in $d$ dimensions:
     $$L(w) = L_{\text{global}}(w) + L_{\text{needle}}(w)$$
     where $L_{\text{global}}$ has a wide, low-curvature minimum at $w_{\text{flat}}^*$, and $L_{\text{needle}}$ embeds a narrow, high-curvature potential well of radius $r_0$ and depth $V_0$ at $w_{\text{sharp}}^*$.
   - Simulate parameter trajectories under standard Gradient Descent, Raw Anti-SAM ($\rho > 0$), and Carve ($\rho > 0$, isochoric gauge).
2. **Trajectory & Needle Entrapment Analysis**:
   - Track distance $d(w(t), w_{\text{sharp}}^*)$ from the needle attractor throughout optimization.
   - Evaluate whether Raw Anti-SAM falls into the needle basin ($d < r_0$) due to morphological erosion basin dilation $r_{\text{eff}} = r_0 + \rho$.
   - Verify whether Carve escapes the needle ($d \gg r_0$) and reaches the flat generalizable basin.
   - Measure the final validation loss ratio $L_{\text{iso}} / L_{\text{anti}}$ on perturbed test distributions.
3. **Transverse Divergence Scaling Diagnostic**:
   - Numerically compute the vector field divergence $\operatorname{div}(E)$ across parameter dimensions $d \in [2, 100]$ using unbiased Hutchinson trace estimators:
     $$\operatorname{div}(E) \approx \frac{1}{M} \sum_{m=1}^M v_m^T J_E(w) v_m, \quad v_m \sim \mathcal{N}(0, I)$$
   - Verify the theoretical scaling law:
     - Raw Anti-SAM: $\operatorname{div}(E_{\text{anti}}) \propto -d \cdot \frac{\operatorname{Tr}_{T^\perp}(H)}{\|\nabla L\|}$ (divergence grows negatively with dimension).
     - Carve: $\operatorname{div}_{T^\perp}(E_{\text{iso}}) = 0$ identically across all dimensions $d$.
4. **Diagnostic Visualizations & Reproducibility**:
   - Output high-resolution diagnostic plots under `analysis/figures/`:
     - `simulation_landscape.png`: Trajectory convergence curves and distance to needle center.
     - `divergence_scaling.png`: Negative divergence scaling of Anti-SAM vs zero divergence of Carve.
   - Maintain fully deterministic seeds (`seed=42`).
5. **Specialized Skill Consultation**:
   - For high-dimensional simulation design and divergence measurement: activate and consult `skills/experimental-research` (`references/simulation-and-measurement.md`, `references/study-design.md`).
   - For caustic curvature and transverse manifold dynamics: activate and consult `skills/theory-research` (`references/proof-audit.md`).
   - *Protocol Rule*: Use and apply these skills internally whenever needed, but do NOT cite skill names, meta-instructions, or tool invocations in final reports or scientific outputs.

## 3. Gate Criteria
Phase 2 passes only if:
- Standard Anti-SAM is trapped in the sharp needle ($d < r_0$), confirming the caustic collapse failure mode.
- Carve escapes the sharp needle attractor ($d \gg r_0$) and converges to the flat minimum.
- Final validation loss ratio satisfies $L_{\text{iso}} / L_{\text{anti}} \le 0.80$.
- Divergence scaling confirms $\operatorname{div}(E_{\text{anti}}) \propto -d$ and $\operatorname{div}(E_{\text{iso}}) = 0$.
- Standard Phase 2 artifacts (`report.md`, `manifest.json`, `commands.log`, `phases/status/phase2.json`) are written.
