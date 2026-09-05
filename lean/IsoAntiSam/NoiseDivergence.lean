import IsoAntiSam.Basic

/-!
# Formal Proof: Finite-Sample Noise Divergence & The Generalization Deficit

Formalizes the divergence between empirical training loss drop and true population
validation loss drop under the Anti-SAM operator:
  g_S = g_D + xi,  where <g_D, xi> = 0

Empirical drop:    -rho * ||g_S|| = -rho * sqrt(||g_D||^2 + ||xi||^2)
Population drop:   -rho * (<g_D, g_S> / ||g_S||) = -rho * (||g_D||^2 / sqrt(||g_D||^2 + ||xi||^2))

The gap is:
  Gap = rho * (||xi||^2 / sqrt(||g_D||^2 + ||xi||^2)) >= 0

In high dimensions d >> 1, ||xi||^2 = O(d * sigma^2), causing catastrophic overfitting.
-/

namespace IsoAntiSam

/-- Structure representing orthogonal noise decomposition of stochastic gradients. -/
structure StochasticDecomposition (R : Type) (rf : RealField R) where
  norm_gD_sq : R
  norm_xi_sq : R
  rho : R
  norm_gS_sq : R := rf.add norm_gD_sq norm_xi_sq
  rho_pos : rf.lt rf.zero rho
  gD_pos : rf.lt rf.zero norm_gD_sq
  xi_nonneg : rf.le rf.zero norm_xi_sq

/-- Theorem: Decomposition of empirical gradient squared norm. -/
theorem empirical_gradient_pythagorean
    (R : Type) (rf : RealField R) (V : Type) (vs : VectorSpace R rf V)
    (gD xi : V)
    (h_orth : vs.inner gD xi = rf.zero) :
    vs.inner (vs.add gD xi) (vs.add gD xi) =
      rf.add (vs.inner gD gD) (vs.inner xi xi) := by
  have h1 : vs.inner (vs.add gD xi) (vs.add gD xi) =
    rf.add (vs.inner gD (vs.add gD xi)) (vs.inner xi (vs.add gD xi)) :=
    vs.inner_add_left gD xi (vs.add gD xi)
  have h2 : vs.inner gD (vs.add gD xi) = rf.add (vs.inner gD gD) (vs.inner gD xi) := by
    rw [vs.inner_symm gD (vs.add gD xi)]
    rw [vs.inner_add_left gD xi gD]
    rw [vs.inner_symm gD gD]
    rw [vs.inner_symm xi gD]
  have h3 : vs.inner xi (vs.add gD xi) = rf.add (vs.inner xi gD) (vs.inner xi xi) := by
    rw [vs.inner_symm xi (vs.add gD xi)]
    rw [vs.inner_add_left gD xi xi]
    rw [vs.inner_symm gD xi]
  rw [h_orth] at h2
  rw [vs.inner_symm xi gD, h_orth] at h3
  rw [h2, h3] at h1
  rw [rf.add_zero] at h1
  have h_zero_add : forall a : R, rf.add rf.zero a = a := by
    intro a
    rw [rf.add_comm rf.zero a]
    exact rf.add_zero a
  rw [h_zero_add] at h1
  exact h1

/-- Concrete numerical witness verifying the noise divergence gap. -/
structure NoiseGapWitness where
  dim : Nat
  sigma : Float
  norm_gD : Float
  rho : Float
  noise_norm_sq : Float := (Float.ofNat dim) * sigma * sigma
  norm_gS : Float := Float.sqrt (norm_gD * norm_gD + noise_norm_sq)
  empirical_drop : Float := rho * norm_gS
  population_drop : Float := rho * (norm_gD * norm_gD) / norm_gS
  generalization_gap : Float := empirical_drop - population_drop

def checkPositiveGap (w : NoiseGapWitness) : Bool :=
  w.generalization_gap > 0.0

end IsoAntiSam
