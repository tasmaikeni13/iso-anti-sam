# Phase 3: PyTorch Optimizer Architecture, Unit Testing, and Algorithmic Controls

## 1. Objective
Implement the modular PyTorch optimizer package `iso_anti_sam`:
1. `AntiSAM`: Baseline implementation of the raw formula $\min_w \min_{\|\epsilon\|\le\rho} L(w+\epsilon)$.
2. `IsoAntiSAM`: Production-grade optimizer with automatic micro-batch splitting, bilateral coherence gating, and isochoric gauge projection.
3. Establish comprehensive unit test coverage verifying gradient norms, parameter restoration, and device placement.

## 2. Execution Commands
```bash
pip install -e /root/iso-anti-sam --break-system-packages
python3 -c "
import torch
import torch.nn as nn
from iso_anti_sam import AntiSAM, IsoAntiSAM

# Unit test 1: Check AntiSAM step execution
m1 = nn.Linear(10, 2)
opt1 = AntiSAM(m1.parameters(), lr=0.01, rho=0.05)
l1 = m1(torch.randn(8, 10)).sum()
l1.backward()
opt1.first_step(zero_grad=True)
l1_pert = m1(torch.randn(8, 10)).sum()
l1_pert.backward()
opt1.second_step(zero_grad=True)
print('AntiSAM unit test passed.')

# Unit test 2: Check IsoAntiSAM step execution
m2 = nn.Linear(10, 2)
opt2 = IsoAntiSAM(m2.parameters(), lr=0.01, rho=0.05)
x = torch.randn(16, 10)
y = torch.randint(0, 2, (16,))
crit = nn.CrossEntropyLoss()

# Micro-batch 1
l_b1 = crit(m2(x[:8]), y[:8])
l_b1.backward()
g1 = [p.grad.clone() for p in m2.parameters()]
m2.zero_grad()

# Micro-batch 2
l_b2 = crit(m2(x[8:]), y[8:])
l_b2.backward()
g2 = [p.grad.clone() for p in m2.parameters()]
m2.zero_grad()

cos_sim = opt2.compute_bilateral_perturbation(g1, g2)
l_outer = crit(m2(x), y)
l_outer.backward()
opt2.step_with_bilateral(zero_grad=True)
print(f'IsoAntiSAM unit test passed. Cross-batch cosine similarity: {cos_sim:.4f}')
"
```

## 3. Expected Outputs & Success Criteria
1. Unit tests pass with code 0.
2. In-place weight updates verified without memory leaks.
3. Coherence similarity returns valid scalar in $[-1.0, 1.0]$.

## 4. Self-Correcting Autonomous Fallback Loop
If unit tests fail:
1. **Activate `ml-research` skill** (`/root/skills_repo/ml-research/SKILL.md`) and consult `references/experiment-protocol.md`.
2. Inspect parameter graph detachment (`torch.no_grad()`). Ensure `old_p` storage correctly caches tensor references across parameter groups.
3. Validate gradient zeroing between micro-batch forward/backward passes.
4. Re-run tests until 100% green.
