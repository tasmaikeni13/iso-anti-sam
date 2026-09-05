# Phase 1: Adversarial Novelty, Morphological Erosion Theory & Lean 4 Formal Audit

Work autonomously in the IsoAntiSAM repository and complete Phase 1. Read `phases/README.md` first and adhere strictly to its state machine and guardrails. This is a foundational mathematical and novelty phase: do not train neural networks or spend significant GPU compute.

## 1. Objective
Subject the core theoretical claims of IsoAntiSAM to hostile mathematical scrutiny. Determine whether **Morphological Erosion Optimization** $\mathcal{E}_\rho[L](w) = \inf_{\|\epsilon\| \le \rho} L(w + \epsilon)$, the **Isochoric Gauge Condition** ($\operatorname{div}_{T^\perp}(E_{\text{iso}}) = 0$), and **Bilateral Coherence Gating** ($\mathbb{E}[\langle g_1, g_2 \rangle] = \|\nabla L_{\mathcal{D}}\|^2$) represent a genuine, non-compositional mathematical primitive, prove all defining properties, and machine-verify the complete formal foundation in Lean 4.

## 2. Required Work
1. **Hostile Mathematical Audit**:
   - Re-derive the optimal linear erosion perturbation $\epsilon^* = -\rho \frac{\nabla L}{\|\nabla L\|}$ and its accelerated descent velocity $\Delta L = -\rho \|\nabla L\| - \eta \|\nabla L\|^2$.
   - Audit the Hamilton-Jacobi viscosity formulation $\partial_\rho u(w, \rho) + \|\nabla u(w, \rho)\| = 0$ and prove strictly negative erosion rate.
   - Prove the caustic collapse theorem: under positive transverse curvature, the raw Anti-SAM perturbation vector field $E_{\text{anti}}(w) = -\rho \frac{\nabla L}{\|\nabla L\|}$ has strictly negative divergence:
     $$\operatorname{div}(E_{\text{anti}}) = -\frac{\rho}{\|\nabla L\|} \operatorname{Tr}_{T^\perp}(H) < 0$$
     causing phase-space volume contraction and turning sharp-needle minima into dissipative attractors.
   - Prove that the transverse projector $P_w^\perp = I - \frac{\nabla L \nabla L^T}{\|\nabla L\|^2}$ strictly annihilates gradient modes ($P_w^\perp \nabla L = 0$), and that enforcing the isochoric gauge $\operatorname{div}_{T^\perp}(E_{\text{iso}}) = 0$ preserves phase-space volume ($\det J_\Phi = 1$).
   - Prove bilateral noise cancellation: for independent micro-batches $B_1, B_2$ with unbiased gradients $g_1 = g_{\mathcal{D}} + \xi_1, g_2 = g_{\mathcal{D}} + \xi_2$, prove $\mathbb{E}[\langle g_1, g_2 \rangle] = \|g_{\mathcal{D}}\|^2 \ge 0$, quenching sample noise variance.
2. **Adversarial Edge-Case Analysis**:
   - Construct adversarial counterexamples: vanishing gradients ($\|\nabla L\| \to 0$), orthogonal micro-batches ($\langle g_1, g_2 \rangle \le 0$), saddle points with mixed curvature signatures, one-sparse gradients, and extreme dynamic range.
   - Formally specify numerical safeguards (e.g. machine epsilon floor $\epsilon_{\text{floor}} = 10^{-12}$) that preserve the exact projective map without acting as tunable hyperparameter crutches.
3. **Prior Art & Collision Audit**:
   - Conduct a systematic literature search for prior art. Compare defining formulations against:
     - Standard Sharpness-Aware Minimization (SAM, ASAM, GSAM, ESAM, LookSAM, Friendly-SAM).
     - Morphological erosion and Hamilton-Jacobi viscosity methods in optimization.
     - Gradient agreement and coherence methods (PCGrad, CAGrad, Cosine-Similarity gating).
   - Document search queries, dates, and formula-level comparisons. Verify that no prior work combines transverse divergence-free isochoric gauges with bilateral micro-batch erosion.
4. **Machine Verification in Lean 4**:
   - Build and verify all 9 core mathematical modules under `lean/IsoAntiSam/*.lean`.
   - Ensure `lake build` completes cleanly with return code 0, zero `sorry` placeholders, and zero unproven axioms.
5. **Documentation & Research Paper**:
   - Update `paper/paper.tex` and `paper/PAPER.md` with the verified theorems, explicit proof steps, and adversarial boundary analyses.
6. **Specialized Skill Consultation**:
   - For mathematical proof auditing and adversarial stress-testing, activate and consult `skills/theory-research` (`references/proof-audit.md`, `references/attack-protocol.md`).
   - For systematic literature searches and novelty matrices, activate and consult `skills/literature-frontier` (`references/search-protocol.md`, `references/frontier-artifacts.md`).
   - *Protocol Rule*: Use and apply these skills internally whenever needed, but do NOT cite skill names, meta-instructions, or tool invocations in final reports or scientific outputs.

## 3. Gate Criteria
Phase 1 passes only if:
- All 9 Lean 4 formal theorems build cleanly with **0 `sorry`** and **0 unproven axioms**.
- `lake build` executes with return code 0.
- Every mathematical statement has a complete, sound written proof covering boundary and edge cases.
- Literature collision audit confirms the novelty of the isochoric gauge and bilateral erosion formulation.
- Standard Phase 1 artifacts (`report.md`, `manifest.json`, `commands.log`, `phases/status/phase1.json`) are written.
