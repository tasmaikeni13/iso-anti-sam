# Phase 4: Native AMD ROCm/HIP C++ Coherent Erosion Kernel & MI300X Hardware Profiling

Work autonomously in the Carve repository and complete Phase 4. Read `phases/README.md` first and require a validated `PASS` handoff from Phase 3. This phase implements hardware-level acceleration on AMD Instinct MI300X GPUs.

## 1. Objective
Develop a high-performance native AMD ROCm/HIP C++ fused kernel targeting the **AMD Instinct MI300X** architecture (`gfx942`). Eliminate Python overhead by fusing cross-batch inner product reductions ($\langle g_1, g_2 \rangle$, $\|g_1\|^2$, $\|g_2\|^2$, $\|g_{\text{avg}}\|^2$), coherence gate evaluation, and in-place weight perturbation into a single high-throughput GPU kernel.

## 2. Required Work
1. **Hardware Inventory & Environment Verification**:
   - Query and log AMD GPU specifications via `rocm-smi`: verify MI300X VF architecture, HBM3 memory capacity (192–205 GB), compute units, and driver status.
   - Verify compiler toolchain: check `/opt/rocm/bin/hipcc --version` and ROCm 6.3 runtime paths.
2. **Native HIP Fused Kernel Development**:
   - Implement `kernels/coherent_erosion_kernel.hip`:
     - Utilize 64-wide wavefront shuffle instructions (`__shfl_down`) native to AMD GCN/CDNA architectures.
     - Stage 1: Parallel reduction over gradients $g_1, g_2$ to accumulate scalar dot products and squared norms in shared memory and block-level atomics.
     - Stage 2: Scalar gate calculation in FP32: $\operatorname{gate} = \max(0, \langle g_1, g_2 \rangle / (\|g_1\| \|g_2\|))$.
     - Stage 3: Vectorized coalesced write back to parameter memory $w_i \leftarrow w_i - \rho \cdot \operatorname{gate} \cdot \frac{g_{\text{avg}, i}}{\|g_{\text{avg}}\|}$.
3. **Rigorous Numerical Validation**:
   - Build a standalone verification binary `kernels/test_kernel.cpp`:
     - Compare the native HIP GPU output against a 64-bit double-precision CPU golden reference on $N = 1,000,000$ elements.
     - Measure relative error of dot products and vector norms.
     - Measure maximum absolute deviation of perturbed parameter arrays.
4. **Throughput & Latency Benchmarking**:
   - Benchmark kernel execution latency across parameter sizes $N \in [10^5, 10^7]$.
   - Ensure the fused kernel executes in $< 3.0$ ms for $N = 1,000,000$ parameters on MI300X.
   - Compare memory bandwidth against theoretical HBM3 peak.
5. **Specialized Skill Consultation**:
   - For GPU execution profiling, latency benchmarking, and hardware measurement: activate and consult `skills/experimental-research` (`references/simulation-and-measurement.md`).
   - For multi-precision reduction tolerances and ROCm compiler diagnostics: activate and consult `skills/ml-research` (`references/experiment-protocol.md`).
   - *Protocol Rule*: Use and apply these skills internally whenever needed, but do NOT cite skill names, meta-instructions, or tool invocations in final reports or scientific outputs.

## 3. Gate Criteria
Phase 4 passes only if:
- `coherent_erosion_kernel.hip` compiles cleanly with `/opt/rocm/bin/hipcc -O3 --offload-arch=gfx942`.
- Standalone test binary runs without errors, warnings, or ROCm memory faults.
- Kernel execution time for $N = 1,000,000$ parameters is **$< 3.0$ ms** on AMD Instinct MI300X.
- Scalar reduction relative error is **$< 10^{-4}$** vs double-precision CPU reference.
- Parameter array maximum elementwise deviation is **$< 10^{-6}$**.
- Standard Phase 4 artifacts (`report.md`, `manifest.json`, `commands.log`, `phases/status/phase4.json`) are written.
