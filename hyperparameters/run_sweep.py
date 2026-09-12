import os
import sys
import json
import time
import math
import subprocess
import numpy as np

sys.path.insert(0, "/root/carve")
sys.path.insert(0, "/root/carve/src")

SWEEP_DIR = "/root/carve/hyperparameters"
LOG_DIR = "/root/carve/logs/hyperparameters"
DATA_DIR = "/root/carve/data/fineweb_edu"

def run_trial(scale, opt, lr, wd, rho, schedule, steps=30, val_interval=10):
    exp_name = f"sweep_{scale}_{opt}_lr{lr}_wd{wd}_rho{rho}_{schedule}"
    cmd = [
        "python3", os.path.join(SWEEP_DIR, "train_distributed.py"),
        "--scale", scale,
        "--optimizer", opt,
        "--lr", str(lr),
        "--weight_decay", str(wd),
        "--rho", str(rho),
        "--rho_schedule", schedule,
        "--max_steps", str(steps),
        "--val_interval", str(val_interval),
        "--batch_size", "4",
        "--grad_accum_steps", "8",
        "--seq_len", "1024",
        "--data_dir", DATA_DIR,
        "--log_dir", LOG_DIR,
        "--exp_name", exp_name
    ]
    print(f"\n=======================================================")
    print(f"Launching Trial: {exp_name}")
    print(f"Command: {' '.join(cmd)}")
    print(f"=======================================================")
    t0 = time.time()
    res = subprocess.run(cmd, capture_output=True, text=True)
    dur = time.time() - t0

    log_path = os.path.join(LOG_DIR, f"{exp_name}_history.json")
    if os.path.exists(log_path):
        with open(log_path, "r") as f:
            h = json.load(f)
        final_val_loss = h["val_loss"][-1]
        final_val_ppl = h["val_ppl"][-1]
        avg_throughput = np.mean(h["throughput_tokens_per_s"])
        final_lambda_max = h["lambda_max"][-1]
        print(f" Trial Complete in {dur:.1f}s | Val Loss: {final_val_loss:.4f} | Val PPL: {final_val_ppl:.2f} | λ_max: {final_lambda_max:.2f} | {avg_throughput:,.0f} tok/s")
        return {
            "exp_name": exp_name,
            "scale": scale,
            "optimizer": opt,
            "lr": lr,
            "weight_decay": wd,
            "rho": rho,
            "schedule": schedule,
            "val_loss": final_val_loss,
            "val_ppl": final_val_ppl,
            "lambda_max": final_lambda_max,
            "throughput": avg_throughput,
            "duration_s": dur,
            "status": "PASS"
        }
    else:
        print(f" Trial Failed! Stderr:\n{res.stderr[-500:]}")
        return {
            "exp_name": exp_name,
            "scale": scale,
            "optimizer": opt,
            "lr": lr,
            "weight_decay": wd,
            "rho": rho,
            "schedule": schedule,
            "val_loss": float("nan"),
            "val_ppl": float("nan"),
            "lambda_max": float("nan"),
            "throughput": 0.0,
            "duration_s": dur,
            "status": "FAIL",
            "error": res.stderr[-500:]
        }

def main():
    os.makedirs(LOG_DIR, exist_ok=True)
    print("================================================================================")
    print("   Starting Automated Multi-Fidelity Hyperparameter Sweep across MI300X GPU     ")
    print("   Optimizers: AdamW, Vanilla SAM, CARVE                                       ")
    print("   Scales: 125M (3B token protocol) & 350M (7B token protocol)                 ")
    print("================================================================================")

    # 1. Prepare data shard
    from hyperparameters.fineweb_loader import prepare_fineweb_edu_shard
    prepare_fineweb_edu_shard(target_tokens=30_000_000, save_dir=DATA_DIR, seed=42)

    all_trials = []

    # --- 125M Sweep Matrix ---
    print("\n>>> Executing 125M Architecture Calibration Sweeps <<<")
    # AdamW candidates
    for lr in [3e-4, 6e-4, 1e-3]:
        for wd in [0.01]:
            res = run_trial("125m", "adamw", lr, wd, 0.0, "constant", steps=25, val_interval=12)
            all_trials.append(res)

    # Vanilla SAM candidates
    for lr in [6e-4]:
        for rho in [0.01, 0.02, 0.05]:
            res = run_trial("125m", "sam", lr, 0.01, rho, "constant", steps=25, val_interval=12)
            all_trials.append(res)

    # CARVE candidates (evaluating rho_0 and schedule)
    for lr in [6e-4]:
        for rho in [0.01, 0.02, 0.05]:
            for sched in ["cosine", "constant"]:
                res = run_trial("125m", "carve", lr, 0.01, rho, sched, steps=25, val_interval=12)
                all_trials.append(res)

    # --- 350M Sweep Matrix ---
    print("\n>>> Executing 350M Architecture Calibration Sweeps <<<")
    # Pilot checks on 350M architecture
    for opt, lr, rho, sched in [
        ("adamw", 4e-4, 0.0, "constant"),
        ("sam", 4e-4, 0.02, "constant"),
        ("carve", 4e-4, 0.02, "cosine"),
        ("carve", 4e-4, 0.01, "cosine")
    ]:
        res = run_trial("350m", opt, lr, 0.01, rho, sched, steps=15, val_interval=7)
        all_trials.append(res)

    # Save aggregated results
    summary_file = os.path.join(SWEEP_DIR, "sweep_results.json")
    with open(summary_file, "w") as f:
        json.dump(all_trials, f, indent=2)

    print("\n================================================================================")
    print("                    HYPERPARAMETER SWEEP SUMMARY TABLE                         ")
    print("================================================================================")
    print(f"{'Scale':<6} | {'Optimizer':<8} | {'LR':<7} | {'Rho':<6} | {'Schedule':<8} | {'Val PPL':<8} | {'λ_max':<7} | {'Tok/s':<9} | {'Status'}")
    print("-" * 88)
    for t in all_trials:
        ppl_str = f"{t['val_ppl']:.2f}" if not math.isnan(t['val_ppl']) else "N/A"
        lmax_str = f"{t['lambda_max']:.2f}" if not math.isnan(t['lambda_max']) else "N/A"
        print(f"{t['scale']:<6} | {t['optimizer']:<8} | {t['lr']:<7.1e} | {t['rho']:<6.2f} | {t['schedule']:<8} | {ppl_str:<8} | {lmax_str:<7} | {t['throughput']:<9,.0f} | {t['status']}")
    print("================================================================================")
    print(f"Results saved to: {summary_file}")

    # Generate 4-panel visualization
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Panel 1: 125M PPL comparison
    ax1 = axes[0, 0]
    t_125 = [t for t in all_trials if t["scale"] == "125m" and t["status"] == "PASS"]
    labels = [f"{t['optimizer'].upper()}\n(lr={t['lr']:.0e}, ρ={t['rho']})" for t in t_125]
    ppls = [t["val_ppl"] for t in t_125]
    colors = ['#3b82f6' if t['optimizer'] == 'carve' else ('#10b981' if t['optimizer'] == 'adamw' else '#f59e0b') for t in t_125]
    bars = ax1.bar(range(len(t_125)), ppls, color=colors, alpha=0.85, edgecolor='black')
    ax1.set_xticks(range(len(t_125)))
    ax1.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    ax1.set_ylabel("Validation Perplexity (lower is better)")
    ax1.set_title("125M Architecture: Validation Perplexity Comparison")
    ax1.grid(True, linestyle="--", alpha=0.5)

    # Panel 2: Hessian Curvature (lambda_max)
    ax2 = axes[0, 1]
    lmaxs = [t["lambda_max"] for t in t_125]
    ax2.bar(range(len(t_125)), lmaxs, color=colors, alpha=0.85, edgecolor='black')
    ax2.set_xticks(range(len(t_125)))
    ax2.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    ax2.set_ylabel(r"Hessian Spectral Sharpness $\lambda_{\max}$")
    ax2.set_title("Hessian Curvature Suppression (Caustic Stability)")
    ax2.grid(True, linestyle="--", alpha=0.5)

    # Panel 3: Perturbation Sensitivity Curve (Carve vs SAM)
    ax3 = axes[1, 0]
    rhos = [0.01, 0.02, 0.05]
    sam_ppls = [next((t["val_ppl"] for t in t_125 if t["optimizer"] == "sam" and abs(t["rho"] - r) < 1e-4), np.nan) for r in rhos]
    carve_cos_ppls = [next((t["val_ppl"] for t in t_125 if t["optimizer"] == "carve" and abs(t["rho"] - r) < 1e-4 and t["schedule"] == "cosine"), np.nan) for r in rhos]
    carve_cst_ppls = [next((t["val_ppl"] for t in t_125 if t["optimizer"] == "carve" and abs(t["rho"] - r) < 1e-4 and t["schedule"] == "constant"), np.nan) for r in rhos]
    ax3.plot(rhos, sam_ppls, 'o--', color='#f59e0b', linewidth=2, label="Vanilla SAM")
    ax3.plot(rhos, carve_cst_ppls, 's--', color='#8b5cf6', linewidth=2, label="CARVE (Constant ρ)")
    ax3.plot(rhos, carve_cos_ppls, '^-', color='#3b82f6', linewidth=2.5, label="CARVE (Cosine Decay Schedule)")
    ax3.set_xlabel(r"Perturbation Radius $\rho_0$")
    ax3.set_ylabel("Validation Perplexity")
    ax3.set_title(r"Perturbation Radius $\rho_0$ Sensitivity on FineWeb-Edu")
    ax3.legend()
    ax3.grid(True, linestyle="--", alpha=0.5)

    # Panel 4: 350M Pilot Comparison
    ax4 = axes[1, 1]
    t_350 = [t for t in all_trials if t["scale"] == "350m" and t["status"] == "PASS"]
    labels_350 = [f"{t['optimizer'].upper()}\n(ρ={t['rho']}, {t['schedule']})" for t in t_350]
    ppls_350 = [t["val_ppl"] for t in t_350]
    colors_350 = ['#3b82f6' if t['optimizer'] == 'carve' else ('#10b981' if t['optimizer'] == 'adamw' else '#f59e0b') for t in t_350]
    ax4.bar(range(len(t_350)), ppls_350, color=colors_350, alpha=0.85, edgecolor='black')
    ax4.set_xticks(range(len(t_350)))
    ax4.set_xticklabels(labels_350, rotation=30, ha='right', fontsize=9)
    ax4.set_ylabel("Validation Perplexity")
    ax4.set_title("350M Architecture: Pilot Validation Comparison")
    ax4.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plot_path1 = os.path.join(SWEEP_DIR, "sweep_comparison.png")
    plot_path2 = "/root/carve/analysis/figures/sweep_comparison.png"
    os.makedirs("/root/carve/analysis/figures", exist_ok=True)
    plt.savefig(plot_path1, dpi=300)
    plt.savefig(plot_path2, dpi=300)
    plt.close()
    print(f"Figures saved to:\n  {plot_path1}\n  {plot_path2}")

if __name__ == "__main__":
    main()
