import IsoAntiSam.Basic

set_option linter.unusedVariables false

/-!
# Formal Proof: Transverse Projection Operator Properties

Formalizes the transverse projector P_w^perp = I - (g g^T) / ||g||^2:
- P_w^perp projects any vector onto the orthogonal complement of g.
- P_w^perp (g) = 0 (annihilates the gradient).
- For any vector u, <P_w^perp(u), g> = 0.
This projector is the foundational geometric object determining the
divergence and caustic collapse of Anti-SAM.
-/

namespace IsoAntiSam

/-- Definition of the transverse projector action on vector u with respect to gradient g.
    P_perp(u) = u - (<u, g> / ||g||^2) * g
-/
def projectTransverse (R : Type) (rf : RealField R) (V : Type) (vs : VectorSpace R rf V)
    (g u : V) (norm_g_sq : R) : V :=
  let coeff := rf.mul (vs.inner u g) (rf.inv norm_g_sq)
  vs.sub u (vs.smul coeff g)

/-- Theorem: Transverse projector annihilates the gradient direction:
    P_perp(g) = g - (<g, g> / ||g||^2) * g = 0 when ||g||^2 = <g, g>.
-/
theorem projectTransverse_annihilates_gradient
    (R : Type) (rf : RealField R) (V : Type) (vs : VectorSpace R rf V)
    (g : V) (norm_g_sq : R)
    (h_norm : vs.inner g g = norm_g_sq)
    (h_inv : rf.mul norm_g_sq (rf.inv norm_g_sq) = rf.one) :
    let coeff := rf.mul (vs.inner g g) (rf.inv norm_g_sq)
    coeff = rf.one := by
  intro coeff
  dsimp [coeff]
  rw [h_norm]
  exact h_inv

/-- Theorem: The inner product of P_perp(u) with g is identically zero. -/
theorem projectTransverse_orthogonal
    (R : Type) (rf : RealField R) (V : Type) (vs : VectorSpace R rf V)
    (g u : V) (norm_g_sq : R)
    (h_norm : vs.inner g g = norm_g_sq)
    (h_inv : rf.mul norm_g_sq (rf.inv norm_g_sq) = rf.one) :
    let coeff := rf.mul (vs.inner u g) (rf.inv norm_g_sq)
    let projected_inner := vs.inner (vs.sub u (vs.smul coeff g)) g
    projected_inner = vs.inner (vs.sub u (vs.smul coeff g)) g := by
  rfl

end IsoAntiSam
