#include <hip/hip_runtime.h>
#include <stdio.h>
#include <math.h>

#define WARP_SIZE 64  // AMD wavefront size is 64 on MI300X (CDNA3)
#define BLOCK_SIZE 256
#define NUM_WARPS (BLOCK_SIZE / WARP_SIZE)

// Vectorized block-level reduction kernel for dot product and norms
__global__ void compute_dot_and_norms_kernel(
    const float* __restrict__ g1,
    const float* __restrict__ g2,
    float* __restrict__ dot_out,
    float* __restrict__ norm1_sq_out,
    float* __restrict__ norm2_sq_out,
    float* __restrict__ norm_avg_sq_out,
    size_t n
) {
    __shared__ float s_dot[NUM_WARPS];
    __shared__ float s_n1[NUM_WARPS];
    __shared__ float s_n2[NUM_WARPS];
    __shared__ float s_navg[NUM_WARPS];

    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;

    float local_dot = 0.0f;
    float local_n1 = 0.0f;
    float local_n2 = 0.0f;
    float local_navg = 0.0f;

    // Vectorized float4 memory load for coalesced 128-bit access
    size_t vec_n = n / 4;
    const float4* g1_v = reinterpret_cast<const float4*>(g1);
    const float4* g2_v = reinterpret_cast<const float4*>(g2);

    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < vec_n; i += gridDim.x * blockDim.x) {
        float4 v1 = g1_v[i];
        float4 v2 = g2_v[i];

        float avg_x = 0.5f * (v1.x + v2.x);
        float avg_y = 0.5f * (v1.y + v2.y);
        float avg_z = 0.5f * (v1.z + v2.z);
        float avg_w = 0.5f * (v1.w + v2.w);

        local_dot += (v1.x * v2.x + v1.y * v2.y + v1.z * v2.z + v1.w * v2.w);
        local_n1 += (v1.x * v1.x + v1.y * v1.y + v1.z * v1.z + v1.w * v1.w);
        local_n2 += (v2.x * v2.x + v2.y * v2.y + v2.z * v2.z + v2.w * v2.w);
        local_navg += (avg_x * avg_x + avg_y * avg_y + avg_z * avg_z + avg_w * avg_w);
    }

    // Residual tail handling
    size_t tail_start = vec_n * 4;
    size_t tail_idx = tail_start + blockIdx.x * blockDim.x + threadIdx.x;
    if (tail_idx < n) {
        float v1 = g1[tail_idx];
        float v2 = g2[tail_idx];
        float avg = 0.5f * (v1 + v2);
        local_dot += v1 * v2;
        local_n1 += v1 * v1;
        local_n2 += v2 * v2;
        local_navg += avg * avg;
    }

    // Intra-wavefront reduction using AMD 64-wide shfl_down
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        local_dot += __shfl_down(local_dot, offset, WARP_SIZE);
        local_n1 += __shfl_down(local_n1, offset, WARP_SIZE);
        local_n2 += __shfl_down(local_n2, offset, WARP_SIZE);
        local_navg += __shfl_down(local_navg, offset, WARP_SIZE);
    }

    // Lane 0 of each wavefront stores to shared memory
    if (lane_id == 0) {
        s_dot[warp_id] = local_dot;
        s_n1[warp_id] = local_n1;
        s_n2[warp_id] = local_n2;
        s_navg[warp_id] = local_navg;
    }
    __syncthreads();

    // Block-level reduction: thread 0 sums the wavefronts and performs a single atomicAdd
    if (threadIdx.x == 0) {
        float b_dot = 0.0f, b_n1 = 0.0f, b_n2 = 0.0f, b_navg = 0.0f;
        for (int w = 0; w < NUM_WARPS; ++w) {
            b_dot += s_dot[w];
            b_n1 += s_n1[w];
            b_n2 += s_n2[w];
            b_navg += s_navg[w];
        }
        atomicAdd(dot_out, b_dot);
        atomicAdd(norm1_sq_out, b_n1);
        atomicAdd(norm2_sq_out, b_n2);
        atomicAdd(norm_avg_sq_out, b_navg);
    }
}

// Vectorized coalesced write back to parameter memory
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
    __shared__ float s_scale;
    if (threadIdx.x == 0) {
        float dot = *dot_in;
        float n1 = sqrtf(*norm1_sq_in) + 1e-12f;
        float n2 = sqrtf(*norm2_sq_in) + 1e-12f;
        float n_avg = sqrtf(*norm_avg_sq_in) + 1e-12f;

        float cos_sim = dot / (n1 * n2);
        float gate = cos_sim > coherence_floor ? cos_sim : coherence_floor;
        s_scale = -rho * gate / n_avg;
    }
    __syncthreads();

    float scale = s_scale;
    size_t vec_n = n / 4;
    float4* w_v = reinterpret_cast<float4*>(w);
    const float4* g1_v = reinterpret_cast<const float4*>(g1);
    const float4* g2_v = reinterpret_cast<const float4*>(g2);

    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < vec_n; i += gridDim.x * blockDim.x) {
        float4 v_w = w_v[i];
        float4 v1 = g1_v[i];
        float4 v2 = g2_v[i];

        float4 res;
        res.x = v_w.x + scale * (0.5f * (v1.x + v2.x));
        res.y = v_w.y + scale * (0.5f * (v1.y + v2.y));
        res.z = v_w.z + scale * (0.5f * (v1.z + v2.z));
        res.w = v_w.w + scale * (0.5f * (v1.w + v2.w));

        w_v[i] = res;
    }

    // Residual tail handling
    size_t tail_start = vec_n * 4;
    size_t tail_idx = tail_start + blockIdx.x * blockDim.x + threadIdx.x;
    if (tail_idx < n) {
        float g_avg = 0.5f * (g1[tail_idx] + g2[tail_idx]);
        w[tail_idx] += scale * g_avg;
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
    (void)hipMemsetAsync(d_scalars, 0, 4 * sizeof(float), stream);

    int blockSize = BLOCK_SIZE;
    // Calculate grid size to fully occupy MI300X Compute Units (304 CUs)
    int numBlocks = (int)((n / 4 + blockSize - 1) / blockSize);
    if (numBlocks > 2048) numBlocks = 2048;
    if (numBlocks < 1) numBlocks = 1;

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
