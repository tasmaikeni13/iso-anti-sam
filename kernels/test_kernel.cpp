#include <hip/hip_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>

#define HIP_CHECK(cmd) do { \
    hipError_t err = cmd; \
    if (err != hipSuccess) { \
        fprintf(stderr, "HIP error %s:%d: '%s'\n", __FILE__, __LINE__, hipGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

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
    printf("=== Testing IsoAntiSAM Fused HIP Kernel on AMD Instinct MI300X ===\n");
    size_t n = 1000000; // 1M parameters
    size_t bytes = n * sizeof(float);

    float *h_w = (float*)malloc(bytes);
    float *h_w_ref = (float*)malloc(bytes);
    float *h_g1 = (float*)malloc(bytes);
    float *h_g2 = (float*)malloc(bytes);

    double cpu_dot = 0.0;
    double cpu_n1_sq = 0.0;
    double cpu_n2_sq = 0.0;
    double cpu_navg_sq = 0.0;

    for (size_t i = 0; i < n; i++) {
        h_w[i] = 1.0f;
        h_w_ref[i] = 1.0f;
        float g1 = 0.05f + 0.01f * sinf((float)i);
        float g2 = 0.05f + 0.01f * cosf((float)i);
        h_g1[i] = g1;
        h_g2[i] = g2;

        float g_avg = 0.5f * (g1 + g2);
        cpu_dot += (double)g1 * (double)g2;
        cpu_n1_sq += (double)g1 * (double)g1;
        cpu_n2_sq += (double)g2 * (double)g2;
        cpu_navg_sq += (double)g_avg * (double)g_avg;
    }

    float rho = 0.05f;
    float coherence_floor = 0.0f;

    // CPU reference perturbation
    double norm1 = sqrt(cpu_n1_sq) + 1e-12;
    double norm2 = sqrt(cpu_n2_sq) + 1e-12;
    double cos_sim = cpu_dot / (norm1 * norm2);
    double gate = cos_sim > coherence_floor ? cos_sim : coherence_floor;
    double norm_avg = sqrt(cpu_navg_sq) + 1e-12;
    double scale = -rho * gate / norm_avg;

    for (size_t i = 0; i < n; i++) {
        float g_avg = 0.5f * (h_g1[i] + h_g2[i]);
        h_w_ref[i] += (float)(scale * g_avg);
    }

    float *d_w, *d_g1, *d_g2, *d_scalars;
    HIP_CHECK(hipMalloc(&d_w, bytes));
    HIP_CHECK(hipMalloc(&d_g1, bytes));
    HIP_CHECK(hipMalloc(&d_g2, bytes));
    HIP_CHECK(hipMalloc(&d_scalars, 4 * sizeof(float)));

    HIP_CHECK(hipMemcpy(d_w, h_w, bytes, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_g1, h_g1, bytes, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_g2, h_g2, bytes, hipMemcpyHostToDevice));

    // Warmup iterations
    for (int it = 0; it < 5; it++) {
        launch_coherent_erosion_mi300x(d_w, d_g1, d_g2, d_scalars, rho, coherence_floor, n, 0);
    }
    HIP_CHECK(hipDeviceSynchronize());

    // Reset d_w for exact golden verification
    HIP_CHECK(hipMemcpy(d_w, h_w, bytes, hipMemcpyHostToDevice));

    hipEvent_t start, stop;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));

    // Timed steady-state execution
    int num_trials = 50;
    HIP_CHECK(hipEventRecord(start, 0));
    for (int it = 0; it < num_trials; it++) {
        launch_coherent_erosion_mi300x(d_w, d_g1, d_g2, d_scalars, rho, coherence_floor, n, 0);
    }
    HIP_CHECK(hipEventRecord(stop, 0));
    HIP_CHECK(hipEventSynchronize(stop));

    float total_ms = 0;
    HIP_CHECK(hipEventElapsedTime(&total_ms, start, stop));
    float ms = total_ms / num_trials;

    // Reset and do single clean run for numerical validation
    HIP_CHECK(hipMemcpy(d_w, h_w, bytes, hipMemcpyHostToDevice));
    launch_coherent_erosion_mi300x(d_w, d_g1, d_g2, d_scalars, rho, coherence_floor, n, 0);
    HIP_CHECK(hipDeviceSynchronize());

    HIP_CHECK(hipMemcpy(h_w, d_w, bytes, hipMemcpyDeviceToHost));
    float h_scalars[4];
    HIP_CHECK(hipMemcpy(h_scalars, d_scalars, 4 * sizeof(float), hipMemcpyDeviceToHost));

    printf("Kernel steady-state latency: %.4f ms for %zu elements (Target < 3.0 ms)\n", ms, n);
    printf("GPU Reductions: Dot: %.6e, N1_sq: %.6e, N2_sq: %.6e, NAvg_sq: %.6e\n",
           h_scalars[0], h_scalars[1], h_scalars[2], h_scalars[3]);
    printf("CPU Reference : Dot: %.6e, N1_sq: %.6e, N2_sq: %.6e, NAvg_sq: %.6e\n",
           cpu_dot, cpu_n1_sq, cpu_n2_sq, cpu_navg_sq);

    // Verify relative error < 1e-4
    float rel_err_dot = fabsf(h_scalars[0] - (float)cpu_dot) / (float)cpu_dot;
    float rel_err_n1 = fabsf(h_scalars[1] - (float)cpu_n1_sq) / (float)cpu_n1_sq;
    float rel_err_n2 = fabsf(h_scalars[2] - (float)cpu_n2_sq) / (float)cpu_n2_sq;
    float rel_err_navg = fabsf(h_scalars[3] - (float)cpu_navg_sq) / (float)cpu_navg_sq;

    printf("Relative Errors: Dot: %.2e, N1: %.2e, N2: %.2e, NAvg: %.2e\n",
           rel_err_dot, rel_err_n1, rel_err_n2, rel_err_navg);

    assert(ms < 3.0f && "Kernel execution time must be < 3.0 ms on MI300X");
    assert(rel_err_dot < 1e-4f && "Dot product relative error exceeds tolerance 1e-4");
    assert(rel_err_n1 < 1e-4f && "Norm1 relative error exceeds tolerance 1e-4");
    assert(rel_err_n2 < 1e-4f && "Norm2 relative error exceeds tolerance 1e-4");
    assert(rel_err_navg < 1e-4f && "NormAvg relative error exceeds tolerance 1e-4");

    // Verify in-place parameter array updates match CPU reference within 1e-6 (strict Phase 4 gate criteria)
    float max_param_diff = 0.0f;
    for (size_t i = 0; i < n; i++) {
        float diff = fabsf(h_w[i] - h_w_ref[i]);
        if (diff > max_param_diff) max_param_diff = diff;
    }
    printf("Max parameter deviation (GPU vs CPU): %.2e\n", max_param_diff);
    assert(max_param_diff < 1e-6f && "Parameter update deviation exceeds tolerance 1e-6");

    // Bandwidth calculation for N = 1,000,000:
    // Reduction: reads g1, g2 (8N bytes)
    // Perturbation: reads w, g1, g2 (12N bytes), writes w (4N bytes)
    // Total DRAM memory traffic = 24N bytes = 24 MB
    double total_bytes = 24.0 * (double)n;
    double gb_transferred = total_bytes / 1e9;
    double effective_bw_gbs = gb_transferred / ((double)ms / 1000.0);
    printf("MI300X Effective Memory Bandwidth (N=1M): %.2f GB/s (Peak HBM3: 5,300 GB/s)\n", effective_bw_gbs);

    printf("ALL REDUCTION AND NUMERICAL ASSERTIONS PASSED ON MI300X.\n");

    HIP_CHECK(hipFree(d_w));
    HIP_CHECK(hipFree(d_g1));
    HIP_CHECK(hipFree(d_g2));
    free(h_w);
    free(h_w_ref);
    free(h_g1);
    free(h_g2);

    // Scaling Benchmark across N in [10^5, 10^7]
    printf("\n=== Parameter Scaling & HBM3 Bandwidth Sweep [10^5, 10^7] ===\n");
    size_t test_sizes[] = {100000, 500000, 1000000, 5000000, 10000000};
    for (int s = 0; s < 5; s++) {
        size_t cur_n = test_sizes[s];
        size_t cur_bytes = cur_n * sizeof(float);
        float *cur_w, *cur_g1, *cur_g2;
        HIP_CHECK(hipMalloc(&cur_w, cur_bytes));
        HIP_CHECK(hipMalloc(&cur_g1, cur_bytes));
        HIP_CHECK(hipMalloc(&cur_g2, cur_bytes));
        HIP_CHECK(hipMemset(cur_w, 0, cur_bytes));
        HIP_CHECK(hipMemset(cur_g1, 0, cur_bytes));
        HIP_CHECK(hipMemset(cur_g2, 0, cur_bytes));

        // Warmup
        for (int it = 0; it < 3; it++) {
            launch_coherent_erosion_mi300x(cur_w, cur_g1, cur_g2, d_scalars, rho, coherence_floor, cur_n, 0);
        }
        HIP_CHECK(hipDeviceSynchronize());

        int cur_trials = (cur_n <= 1000000) ? 50 : 20;
        HIP_CHECK(hipEventRecord(start, 0));
        for (int it = 0; it < cur_trials; it++) {
            launch_coherent_erosion_mi300x(cur_w, cur_g1, cur_g2, d_scalars, rho, coherence_floor, cur_n, 0);
        }
        HIP_CHECK(hipEventRecord(stop, 0));
        HIP_CHECK(hipEventSynchronize(stop));

        float cur_ms = 0;
        HIP_CHECK(hipEventElapsedTime(&cur_ms, start, stop));
        cur_ms /= cur_trials;

        double cur_traffic_bytes = 24.0 * (double)cur_n;
        double cur_bw_gbs = (cur_traffic_bytes / 1e9) / ((double)cur_ms / 1000.0);
        printf("  N = %10zu | Latency: %7.4f ms | Effective Bandwidth: %7.2f GB/s\n",
               cur_n, cur_ms, cur_bw_gbs);

        HIP_CHECK(hipFree(cur_w));
        HIP_CHECK(hipFree(cur_g1));
        HIP_CHECK(hipFree(cur_g2));
    }

    HIP_CHECK(hipFree(d_scalars));
    HIP_CHECK(hipEventDestroy(start));
    HIP_CHECK(hipEventDestroy(stop));

    printf("\nKernel test and scaling sweep completed successfully on AMD Instinct MI300X!\n");
    return 0;
}
