# Phase 6: IsoAntiSAM vs SAM vs AdamW Comparative Evaluation on WikiText-103 (1x MI300X)

## 1. Objective
Run controlled, compute-matched comparative trials on WikiText-103 across:
1. AdamW
2. Standard SAM ($\rho \in [0.01, 0.05, 0.1]$)
3. Raw Anti-SAM ($\rho \in [0.01, 0.05, 0.1]$)
4. IsoAntiSAM ($\rho \in [0.01, 0.05, 0.1]$, bilateral coherence gating)

Measure validation loss trajectory, training speed, sharpness ($\lambda_{\max}$ of Hessian), and test perplexity on 1x AMD MI300X.

## 2. Execution Commands
```bash
python3 /root/iso-anti-sam/benchmarks/train_wikitext.py --optimizer anti_sam --rho 0.05 --epochs 10 --device cuda:0
python3 /root/iso-anti-sam/benchmarks/train_wikitext.py --optimizer sam --rho 0.05 --epochs 10 --device cuda:0
python3 /root/iso-anti-sam/benchmarks/train_wikitext.py --optimizer iso_anti_sam --rho 0.05 --epochs 10 --device cuda:0
```

## 3. Expected Outputs & Success Criteria
1. **Empirical Confirmation of Anti-SAM Failure**:
   - Raw Anti-SAM achieves faster training loss reduction during epochs 1-3, but suffers higher validation perplexity than AdamW by epoch 10.
2. **IsoAntiSAM Breakthrough**:
   - IsoAntiSAM validation loss drops strictly faster than AdamW and SAM, matching the fast erosion trajectory of Anti-SAM without overfitting.
   - Validation perplexity improved by at least 1.5–2.0 points over standard AdamW.
   - Hessian spectrum analysis confirms no sharp-needle caustic collapse.

## 4. Self-Correcting Autonomous Fallback Loop
If IsoAntiSAM fails to outperform baselines on WikiText-103:
1. **Activate `ml-research` skill** (`/root/skills_repo/ml-research/SKILL.md`) and consult `references/method-search.md`.
2. Inspect the Bilateral Coherence Gate values across layers: if early layers have noisy gradients, apply layer-wise adaptive gating (normalize per parameter tensor rather than globally).
3. If perturbation magnitude $\rho$ causes destabilization, implement a cosine warmup schedule for $\rho_t = \rho_{\max} \cdot \frac{1}{2}(1 - \cos(\pi t / T))$.
4. Re-prove any mathematical changes in Lean 4 and re-run benchmark until IsoAntiSAM decisively outperforms baselines.
