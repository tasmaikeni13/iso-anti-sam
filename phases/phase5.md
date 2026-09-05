# Phase 5: Small NLP Benchmark: WikiText-103 Baseline Setup on 1x AMD MI300X

## 1. Objective
Establish a reproducible, clean baseline benchmark on a standard small NLP task (WikiText-103 language modeling) using a causal Transformer (GPT-2 style architecture) on 1x AMD Instinct MI300X accelerator.

## 2. Research Contract
- Task: Autoregressive Language Modeling (WikiText-103).
- Architecture: 6-layer causal Transformer (dim=384, heads=6, seq_len=256, ~15M parameters).
- Metric: Perplexity (PPL) and Cross-Entropy Validation Loss.
- Hardware: 1x AMD Instinct MI300X (192GB VRAM).
- Baseline optimizers: Standard SGD with momentum, AdamW.

## 3. Execution Commands
```bash
python3 -c "
import urllib.request
import os

data_dir = '/root/iso-anti-sam/data/wikitext103'
os.makedirs(data_dir, exist_ok=True)
print('Setting up WikiText-103 data pipeline...')
"
python3 /root/iso-anti-sam/benchmarks/train_wikitext.py --optimizer adamw --epochs 5 --device cuda:0
```

## 4. Expected Outputs & Success Criteria
1. Baseline training completes without out-of-memory (OOM) or NaNs.
2. AdamW baseline establishes reference validation perplexity.
3. Checkpoints and loss curves logged in `/root/iso-anti-sam/logs/wikitext/`.

## 5. Self-Correcting Autonomous Fallback Loop
If training crashes or baseline perplexity is poor:
1. **Activate `ml-research` skill** (`/root/skills_repo/ml-research/SKILL.md`) and consult `references/experiment-protocol.md`.
2. Check tokenization and vocabulary offsets (ensure pad and EOS tokens do not leak into loss calculation).
3. Check learning rate schedule (cosine warmup vs linear decay).
4. Verify ROCm PyTorch memory management: set `PYTORCH_HIP_ALLOC_CONF=garbage_collection_threshold:0.8,max_split_size_mb:512`.
5. Repeat until baseline runs cleanly and records verified loss curve.
