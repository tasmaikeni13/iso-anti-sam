# Carve: Machine-Verified Formal Proofs in Lean 4

This directory contains formal, machine-checked Lean 4 (v4.33.1) proofs of the 9 foundational theorems for the **Carve** optimizer.

## Modules
- `Carve/Basic.lean`: Real field, inner product spaces, and Hamilton-Jacobi erosion definitions.
- `Carve/ErosionVelocity.lean`: Optimal linear descent inner product and velocity advantage.
- `Carve/CausticCollapse.lean`: Phase-space volume contraction and isochoric gauge conservation.
- `Carve/ProjectorProperties.lean`: Transverse projector isolation of non-gradient modes.
- `Carve/HamiltonJacobiErosion.lean`: Hamilton-Jacobi erosion viscosity flow.
- `Carve/BasinDilation.lean`: Catchment basin dilation under morphological erosion.
- `Carve/NoiseDivergence.lean`: Finite-sample noise divergence gap.
- `Carve/CoherentGeneralization.lean`: Cross-batch bilateral noise cancellation theorem.
- `Carve/CoherentDispersion.lean`: Bilateral consensus noise variance reduction.

## Build and Run
```bash
lake clean
lake build
./.lake/build/bin/carve
```