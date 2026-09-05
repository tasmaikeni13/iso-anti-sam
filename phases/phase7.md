# Phase 7: FineWeb-Edu Data Pipeline, Fast Tokenization, and Distributed Infrastructure Setup

## 1. Objective
Establish an enterprise-grade, high-throughput data loading and tokenization pipeline for the HuggingFace `HuggingFaceFW/fineweb-edu` dataset.
Set up multi-GPU DistributedDataParallel (PyTorch DDP / ROCm RCCL) communication across 8x AMD Instinct MI300X GPUs.

## 2. Infrastructure & Data Specifications
- Dataset: FineWeb-Edu (`sample-10BT` or streamed shards).
- Tokenizer: GPT-Neox / Llama / Tiktoken Byte-Pair Encoding (vocab size ~32,000 or 50,257).
- Target sequence length: 2048 tokens.
- Distributed backend: PyTorch with AMD RCCL (`torch.distributed.init_process_group('nccl')`).
- Hardware: Up to 8x AMD Instinct MI300X (total 1.536 TB HBM3 memory).

## 3. Execution Commands
```bash
python3 -c "
import torch
import torch.distributed as dist
print('ROCm visible devices:', torch.cuda.device_count())
"
python3 /root/iso-anti-sam/data/prepare_fineweb.py --sample-tokens 1000000000 --out-dir /root/iso-anti-sam/data/fineweb_1B
```

## 4. Expected Outputs & Success Criteria
1. Tokenization throughput > 50,000 tokens/sec per CPU worker.
2. Verified binary shard format (`.bin` / `.mmap`) for zero-copy memory-mapped disk access.
3. Multi-GPU ring all-reduce communication verified across all MI300X nodes without RCCL timeouts.

## 5. Self-Correcting Autonomous Fallback Loop
If streaming or distributed setup stalls:
1. **Activate `experimental-research` skill** (`/root/skills_repo/experimental-research/SKILL.md`) and consult `references/simulation-and-measurement.md`.
2. Check RCCL environment variables: export `NCCL_DEBUG=INFO`, `HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7`.
3. If internet throughput throttles token downloading, switch to cached streaming with local chunking.
4. Verify token offsets and attention masks. Iterate until multi-GPU pipeline is 100% verified.
