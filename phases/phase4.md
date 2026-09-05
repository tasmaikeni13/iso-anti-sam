# Phase 4: Native HIP C++ Kernel Development and GPU Profiling on AMD Instinct MI300X

## 1. Objective
Develop, optimize, and profile native AMD ROCm/HIP C++ kernels for fused Bilateral Coherence reduction and parameter perturbation on 1x AMD Instinct MI300X (`gfx942`).
Eliminate Python CPU-GPU synchronization bottlenecks by performing cross-batch inner product reductions and in-place tensor additions entirely within AMD wavefronts (64 threads).

## 2. Hardware Environment
- Accelerator: 1x AMD Instinct MI300X (750W, 192GB HBM3, `gfx942`)
- Compiler: `/opt/rocm/bin/hipcc` (ROCm 6.3 / HIP 7.15)

## 3. Execution Commands
```bash
cd /root/iso-anti-sam/kernels
/opt/rocm/bin/hipcc -O3 --offload-arch=gfx942 \
  coherent_erosion_kernel.hip test_kernel.cpp \
  -o test_kernel
./test_kernel
```

## 4. Expected Outputs & Success Criteria
1. Clean compilation with returncode 0.
2. Kernel execution runtime on 1,000,000 elements < 3.0 ms on MI300X.
3. Verification that scalar reductions (`dot`, `norm1_sq`, `norm2_sq`, `norm_avg_sq`) match host CPU calculations within relative tolerance $10^{-4}$.
4. Parameter array correctly updated in-place on device memory.

## 5. Self-Correcting Autonomous Fallback Loop
If HIP kernel fails or performance degrades:
1. **Activate `experimental-research` skill** (`/root/skills_repo/experimental-research/SKILL.md`) and consult `references/simulation-and-measurement.md`.
2. Check architecture target: ensure `--offload-arch=gfx942` is specified.
3. Inspect wavefront shuffle intrinsics: verify `__shfl_down` operates across 64-lane boundaries.
4. If atomic collisions occur during multi-block reduction, transition to a two-pass parallel reduction tree.
5. Benchmark with `rocprof` or `hipEvent` until execution time and accuracy targets are satisfied.
