#include <hip/hip_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

extern "C" void launch_coherent_erosion_mi300x(
    float* d_w,
    const float* d_g1,
    const float* d_g2,
    float* d_scalars,
    float rho,
    float coherence_floor,
    size_t n,
    hipStream_t stream
);

int main() {
    printf("=== Testing IsoAntiSAM Fused HIP Kernel on AMD MI300X ===\n");
    size_t n = 1000000; // 1M parameters
    size_t bytes = n * sizeof(float);

    float *h_w = (float*)malloc(bytes);
    float *h_g1 = (float*)malloc(bytes);
    float *h_g2 = (float*)malloc(bytes);

    for (size_t i = 0; i < n; i++) {
        h_w[i] = 1.0f;
        h_g1[i] = 0.05f + 0.01f * sinf(i);
        h_g2[i] = 0.05f + 0.01f * cosf(i);
    }

    float *d_w, *d_g1, *d_g2, *d_scalars;
    hipMalloc(&d_w, bytes);
    hipMalloc(&d_g1, bytes);
    hipMalloc(&d_g2, bytes);
    hipMalloc(&d_scalars, 4 * sizeof(float));

    hipMemcpy(d_w, h_w, bytes, hipMemcpyHostToDevice);
    hipMemcpy(d_g1, h_g1, bytes, hipMemcpyHostToDevice);
    hipMemcpy(d_g2, h_g2, bytes, hipMemcpyHostToDevice);

    float rho = 0.05f;
    float coherence_floor = 0.0f;

    hipEvent_t start, stop;
    hipEventCreate(&start);
    hipEventCreate(&stop);

    hipEventRecord(start, 0);
    launch_coherent_erosion_mi300x(d_w, d_g1, d_g2, d_scalars, rho, coherence_floor, n, 0);
    hipEventRecord(stop, 0);
    hipEventSynchronize(stop);

    float ms = 0;
    hipEventElapsedTime(&ms, start, stop);

    hipMemcpy(h_w, d_w, bytes, hipMemcpyDeviceToHost);
    float h_scalars[4];
    hipMemcpy(h_scalars, d_scalars, 4 * sizeof(float), hipMemcpyDeviceToHost);

    printf("Kernel executed in: %.3f ms for %zu elements\n", ms, n);
    printf("Scalar Dot: %.4e, Norm1_sq: %.4e, Norm2_sq: %.4e, NormAvg_sq: %.4e\n",
           h_scalars[0], h_scalars[1], h_scalars[2], h_scalars[3]);
    printf("Sample w[0] after perturbation: %.6f (was 1.000000)\n", h_w[0]);

    hipFree(d_w);
    hipFree(d_g1);
    hipFree(d_g2);
    hipFree(d_scalars);
    free(h_w);
    free(h_g1);
    free(h_g2);

    printf("Kernel test completed successfully on AMD MI300X!\n");
    return 0;
}
