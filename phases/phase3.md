# Phase 3: PyTorch Optimizer Architecture, Unit Testing & Algorithmic Invariants

Work autonomously in the Carve repository and complete Phase 3. Read `phases/README.md` first and require a validated `PASS` handoff from Phase 2. Implement the mathematical specification exactly without introducing unauthorized momentum or heuristic rescues.

## 1. Objective
Develop a production-grade, cleanly packaged PyTorch implementation of `Carve`. Establish computational graph isolation, detached gradient computation, in-place weight restoration, state-free parameter memory, and an exhaustive unit test suite.

## 2. Required Work
1. **PyTorch Optimizer Architecture**:
   - Implement `Carve` in `src/carve/carve.py` as a subclass of `torch.optim.Optimizer`.
   - Wrap an underlying base optimizer (e.g. `torch.optim.AdamW` or `torch.optim.SGD`).
   - Implement the two-phase bilateral stepping protocol:
     1. `compute_bilateral_perturbation(grads_b1, grads_b2)`:
        - Compute cross-batch inner product $\langle g_1, g_2 \rangle$ and norms $\|g_1\|, \|g_2\|$ across all parameter tensors in FP32.
        - Evaluate the coherence cosine similarity $\cos(g_1, g_2) = \frac{\langle g_1, g_2 \rangle}{\|g_1\| \|g_2\| + \epsilon}$.
        - Apply coherence gate: $\operatorname{gate} = \max(\text{coherence\_floor}, \cos(g_1, g_2))$.
        - Compute average gradient $g_{\text{avg}} = \frac{1}{2}(g_1 + g_2)$ and apply in-place perturbation:
          $$w_{\text{pert}} = w - \rho \cdot \operatorname{gate} \cdot \frac{g_{\text{avg}}}{\|g_{\text{avg}}\|}$$
        - Save original weights $w$ in optimizer state.
     2. `step_with_bilateral(zero_grad=True)`:
        - Restore original weights $w$ in-place prior to the base optimizer update.
        - Execute base optimizer step using gradients evaluated at the perturbed position $w_{\text{pert}}$.
        - Zero gradients cleanly.
2. **Algorithmic Invariants & Graph Safety**:
   - Verify zero-grad hygiene: micro-batch backward passes must not retain computational graph history or leak across iterations.
   - Verify that parameter tensors are restored bitwise before the base optimizer step.
   - Ensure that the optimizer persists zero temporal state beyond the standard base optimizer moments.
3. **Comprehensive Unit Testing Suite**:
   - Implement `tests/test_optimizer.py` covering:
     - `test_zero_grad_isolation`: Gradients are properly detached; no autograd graph memory leak.
     - `test_inplace_perturbation_recovery`: Original parameter weights are bitwise restored after `step_with_bilateral`.
     - `test_orthogonal_gradient_quenching`: When $g_1 \perp g_2$ ($\langle g_1, g_2 \rangle \le 0$), the gate collapses to 0, resulting in zero perturbation ($\epsilon = 0$).
     - `test_aligned_gradient_preservation`: When $g_1 = g_2$, $\operatorname{gate} = 1$, delivering the full morphological erosion step.
     - `test_multi_param_group_handling`: Correct handling of disparate tensor shapes, biases, and normalization weights.
4. **Packaging & Installation**:
   - Configure `setup.py` and verify editable installation via `pip install -e .`.
5. **Specialized Skill Consultation**:
   - For optimizer invariant enforcement, gradient detachment, and state-free protocol validation: activate and consult `skills/ml-research` (`references/experiment-protocol.md`, `references/method-search.md`).
   - For adversarial tensor shape and edge-case unit test coverage: activate and consult `skills/theory-research` (`references/attack-protocol.md`).
   - *Protocol Rule*: Use and apply these skills internally whenever needed, but do NOT cite skill names, meta-instructions, or tool invocations in final reports or scientific outputs.

## 3. Gate Criteria
Phase 3 passes only if:
- `pip install -e .` succeeds with clean package metadata.
- 100% of unit tests in `tests/test_optimizer.py` pass without warning or error.
- Verified zero memory leaks in repeated forward-backward stepping.
- State-space footprint contains no persistent optimizer state beyond base optimizer requirements.
- Standard Phase 3 artifacts (`report.md`, `manifest.json`, `commands.log`, `phases/status/phase3.json`) are written.
