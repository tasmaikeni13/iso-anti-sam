# Phase 10: Downstream Evaluations, Ablation Diagnostics, and Comprehensive Research Finalization

## 1. Objective
Execute standard downstream NLP benchmarks on the pretrained 125M and 350M checkpoints (HellaSwag, ARC-Easy, MMLU-stem, PIQA, LAMBADA).
Run comprehensive ablation studies isolating each architectural component of IsoAntiSAM.
Finalize the research paper with empirical findings, publish release artifacts, and prepare the camera-ready submission.

## 2. Benchmark Suite
- Common-sense reasoning: HellaSwag, PIQA, ARC-Easy.
- Language modeling: WikiText-103, LAMBADA.
- Zero-shot evaluation using `lm-evaluation-harness`.

## 3. Execution Commands
```bash
python3 /root/iso-anti-sam/eval/evaluate_downstream.py \
  --checkpoint /root/iso-anti-sam/checkpoints/350M_iso_anti_sam \
  --tasks hellaswag,arc_easy,piqa,lambada \
  --output_file /root/iso-anti-sam/results/downstream_350m.json

python3 /root/iso-anti-sam/eval/run_ablations.py \
  --model_size 125M \
  --ablate coherence_gate,isochoric_gauge \
  --output_dir /root/iso-anti-sam/results/ablations
```

## 4. Expected Outputs & Success Criteria
1. IsoAntiSAM delivers statistically significant gains in zero-shot downstream accuracy (+1.5% to +3.0% over AdamW).
2. Ablation experiments prove:
   - Removing the Bilateral Coherence Gate leads to immediate validation divergence (reproducing Anti-SAM failure).
   - Removing the Isochoric Gauge increases sharpness and degrades test accuracy.
3. All research records, tables, and figures updated in `/root/iso-anti-sam/paper/`.

## 5. Self-Correcting Autonomous Fallback Loop
If downstream evaluation underperforms expectations:
1. **Activate `literature-frontier` skill** (`/root/skills_repo/literature-frontier/SKILL.md`) and consult `references/frontier-artifacts.md`.
2. Compare prompt templates and normalization schemes with official HuggingFace leaderboards to eliminate evaluation artifacts.
3. Attribute any discrepancy using the Claim-Evidence Matrix in `/root/skills_repo/ml-research/references/research-loop.md`.
4. Compile final results into `paper/paper.tex` and generate the final publication PDF.
