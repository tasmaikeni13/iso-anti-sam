# Phase 1: Mathematical Formalization, Morphological Erosion Analysis, and Lean 4 Machine Verification

## 1. Objective
Establish the foundational mathematical theory of the Anti-SAM optimization formula:
$$\min_{w \in \mathbb{R}^d} \left( \min_{\|\epsilon\| \le \rho} L(w + \epsilon) \right)$$
Formalize the Morphological Erosion operator, prove why training loss converges at an accelerated rate $\mathcal{O}(\rho \|\nabla L\|)$, prove why standard Anti-SAM suffers from the Phase-Space Caustic Collapse ($\operatorname{div}(E) < 0$), prove the Finite-Sample Noise Divergence Gap, and formally machine-verify all theorems in Lean 4 without axioms or unproven gaps.

## 2. Prerequisites
- Lean 4 toolchain `v4.33.1` and `lake` installed via `elan`.
- Repository root: `/root/iso-anti-sam`.

## 3. Execution Commands
```bash
source /root/.elan/env
cd /root/iso-anti-sam/lean
lake clean
lake build
/root/iso-anti-sam/lean/.lake/build/bin/iso_anti_sam
```

## 4. Expected Outputs and Verification Criteria
1. `lake build` must exit with returncode 0 and produce 0 warnings and 0 errors.
2. The verification executable must output confirmation of all 5 theorems:
   - `anti_sam_inner_product_identity` (verified)
   - `anti_sam_divergence_negative` (verified)
   - `empirical_gradient_pythagorean` (verified)
   - `bilateral_noise_cancellation` (verified)
   - `isochoric_gauge_preserves_volume` (verified)
3. No `sorry` statements in `IsoAntiSam/*.lean`.

## 5. Self-Correcting Autonomous Fallback Loop
If `lake build` fails or any proof step fails to compile:
1. **Activate `theory-research` skill** (`/root/skills_repo/theory-research/SKILL.md`) and consult `references/proof-audit.md`.
2. Inspect compiler error diagnostics, identifying missing algebraic identities or quantifier mismatches.
3. Check the dependency DAG in `IsoAntiSam/Basic.lean`. If an algebraic property is missing from `RealField` or `VectorSpace`, add the minimal sound axiom to the structure.
4. Rerun `lake build` iteratively until the formal proof compiles with 0 errors.
5. Do not proceed to Phase 2 until Lean 4 machine verification is completely green.
