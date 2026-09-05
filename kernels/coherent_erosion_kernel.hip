#include <hip/hip_runtime.h>
#include <stdio.h>
#include <math.h>

#define WARP_SIZE 64  // AMD wavefront size is 64 on MI300X

// Fused kernel to compute dot product and norms of g1, g2 using atomic/block reductions
__global__ void compute_dot_and_norms_kernel(
    const float* __restrict__ g1,
    const float* __restrict__ g2,
    float* __restrict__ dot_out,
    float* __restrict__ norm1_sq_out,
    float* __restrict__ norm2_sq_out,
    float* __restrict__ norm_avg_sq_out,
    size_t n
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    float local_dot = 0.0f;
    float local_n1 = 0.0f;
    float local_n2 = 0.0f;
    float local_navg = 0.0f;

    if (idx < n) {
        float v1 = g1[idx];
        float v2 = g2[idx];
        float v_avg = 0.5f * (v1 + v2);
        local_dot = v1 * v2;
        local_n1 = v1 * v1;
        local_n2 = v2 * v2;
        local_navg = v_avg * v_avg;
    }

    // Wavefront reduction
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        local_dot += __shfl_down(local_dot, offset);
        local_n1 += __shfl_down(local_n1, offset);
        local_n2 += __shfl_down(local_n2, offset);
        local_navg += __shfl_down(local_navg, offset);
    }

    if (threadIdx.x % WARP_SIZE == 0) {
        atomicAdd(dot_out, local_dot);
        atomicAdd(norm1_sq_out, local_n1);
        atomicAdd(norm2_sq_out, local_n2);
        atomicAdd(norm_avg_sq_out, local_navg);
    }
}

// Fused kernel to apply the coherent erosion perturbation to parameters w
__global__ void apply_coherent_erosion_kernel(
    float* __restrict__ w,
    const float* __restrict__ g1,
    const float* __restrict__ g2,
    const float* __restrict__ dot_in,
    const float* __restrict__ norm1_sq_in,
    const float* __restrict__ norm2_sq_in,
    const float* __restrict__ norm_avg_sq_in,
    float rho,
    float coherence_floor,
    size_t n
) {
    float dot = *dot_in;
    float n1 = sqrtf(*norm1_sq_in) + 1e-12f;
    float n2 = sqrtf(*norm2_sq_in) + 1e-12f;
    float n_avg = sqrtf(*norm_avg_sq_in) + 1e-12f;

    float cos_sim = dot / (n1 * n2);
    float gate = cos_sim > coherence_floor ? cos_sim : coherence_floor;
    float scale = -rho * gate / n_avg;

    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float g_avg = 0.5f * (g1[idx] + g2[idx]);
        w[idx] += scale * g_avg;
    }
}

extern "C" void launch_coherent_erosion_mi300x(
    float* d_w,
    const float* d_g1,
    const float* d_g2,
    float* d_scalars, // buffer of size 4: [dot, norm1_sq, norm2_sq, norm_avg_sq]
    float rho,
    float coherence_floor,
    size_t n,
    hipStream_t stream
) {
    hipMemsetAsync(d_scalars, 0, 4 * sizeof(float), stream);
    
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;

    hipLaunchKernelGGL(
        compute_dot_and_norms_kernel,
        dim3(numBlocks), dim3(blockSize), 0, stream,
        d_g1, d_g2,
        &d_scalars[0], &d_scalars[1], &d_scalars[2], &d_scalars[3],
        n
    );

    hipLaunchKernelGGL(
        apply_coherent_erosion_kernel,
        dim3(numBlocks), dim3(blockSize), 0, stream,
        d_w, d_g1, d_g2,
        &d_scalars[0], &d_scalars[1], &d_scalars[2], &d_scalars[3],
        rho, coherence_floor,
        n
    );
}
