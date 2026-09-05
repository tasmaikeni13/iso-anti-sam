import time
import subprocess
import torch

def benchmark_pytorch_mi300x(n=1_000_000, trials=100):
    device = torch.device("cuda:0")
    w = torch.ones(n, device=device, dtype=torch.float32)
    g1 = (0.05 + 0.01 * torch.sin(torch.arange(n, device=device, dtype=torch.float32)))
    g2 = (0.05 + 0.01 * torch.cos(torch.arange(n, device=device, dtype=torch.float32)))
    
    # Warmup
    for _ in range(10):
        dot = torch.dot(g1, g2)
        n1 = torch.linalg.vector_norm(g1)
        n2 = torch.linalg.vector_norm(g2)
        cos_sim = dot / (n1 * n2 + 1e-12)
        gate = torch.clamp(cos_sim, min=0.0)
        g_avg = 0.5 * (g1 + g2)
        n_avg = torch.linalg.vector_norm(g_avg) + 1e-12
        scale = -0.05 * gate / n_avg
        w.add_(g_avg, alpha=scale.item())
    torch.cuda.synchronize()

    # Timing
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    start_event.record()
    for _ in range(trials):
        dot = torch.dot(g1, g2)
        n1 = torch.linalg.vector_norm(g1)
        n2 = torch.linalg.vector_norm(g2)
        cos_sim = dot / (n1 * n2 + 1e-12)
        gate = torch.clamp(cos_sim, min=0.0)
        g_avg = 0.5 * (g1 + g2)
        n_avg = torch.linalg.vector_norm(g_avg) + 1e-12
        scale = -0.05 * gate / n_avg
        w.add_(g_avg, alpha=scale.item())
    end_event.record()
    torch.cuda.synchronize()
    
    total_ms = start_event.elapsed_time(end_event)
    avg_ms = total_ms / trials
    return avg_ms

if __name__ == "__main__":
    print("Benchmarking PyTorch on AMD Instinct MI300X...")
    py_time = benchmark_pytorch_mi300x()
    print(f"PyTorch Multi-Kernel Eager on MI300X: {py_time:.3f} ms / step (1M elements)")
    
    # Run HIP test kernel
    res = subprocess.run(["/root/iso-anti-sam/kernels/test_kernel"], capture_output=True, text=True)
    for line in res.stdout.splitlines():
        if "Kernel executed in:" in line:
            print(f"Native HIP Fused Kernel on MI300X:   {line.split('in: ')[1].split(' for')[0]} / step (1M elements)")
            break
