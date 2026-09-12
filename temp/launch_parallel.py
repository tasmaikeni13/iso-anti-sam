import os
import sys
import time
import math
import json
import subprocess
from datetime import datetime, timezone, timedelta

TEMP_DIR = "/root/carve/temp"
LOG_DIR = os.path.join(TEMP_DIR, "logs")

RUNS = [
    {"optimizer": "adamw", "seed": 42, "lr": 6e-4, "rho": 0.0},
    {"optimizer": "adamw", "seed": 43, "lr": 6e-4, "rho": 0.0},
    {"optimizer": "sam",   "seed": 42, "lr": 6e-4, "rho": 0.02},
    {"optimizer": "sam",   "seed": 43, "lr": 6e-4, "rho": 0.02},
    {"optimizer": "carve", "seed": 42, "lr": 6e-4, "rho": 0.02},
    {"optimizer": "carve", "seed": 43, "lr": 6e-4, "rho": 0.02},
]

def main():
    os.makedirs(LOG_DIR, exist_ok=True)
    print("================================================================================")
    print("     Launching 6 Parallel 125M Pretraining Runs on AMD Instinct MI300X         ")
    print("     Total Budget: 300M tokens per run (1.8 Billion aggregate tokens)           ")
    print("     Optimizers: AdamW (Seeds 42, 43), SAM (Seeds 42, 43), CARVE (Seeds 42, 43) ")
    print("================================================================================", flush=True)

    processes = []
    for run in RUNS:
        opt = run["optimizer"]
        seed = run["seed"]
        lr = run["lr"]
        rho = run["rho"]

        run_id = f"{opt}_seed{seed}"
        log_json = os.path.join(LOG_DIR, f"{run_id}_history.json")
        out_log = os.path.join(LOG_DIR, f"{run_id}.out")

        # Clean previous log files if existing
        if os.path.exists(log_json):
            try: os.remove(log_json)
            except: pass
        if os.path.exists(out_log):
            try: os.remove(out_log)
            except: pass

        cmd = [
            "python3", os.path.join(TEMP_DIR, "train_worker.py"),
            "--optimizer", opt,
            "--seed", str(seed),
            "--lr", str(lr),
            "--rho", str(rho),
            "--total_tokens", "300000000",
            "--batch_size", "8",
            "--grad_accum_steps", "4",
            "--seq_len", "2048",
            "--log_file", log_json
        ]

        env = dict(os.environ)
        env["PYTHONUNBUFFERED"] = "1"

        out_fd = open(out_log, "w")
        p = subprocess.Popen(cmd, stdout=out_fd, stderr=subprocess.STDOUT, env=env)
        processes.append({"run_id": run_id, "pid": p.pid, "p": p, "out_log": out_log, "log_json": log_json})
        print(f"-> Started {run_id:<15} (PID: {p.pid}) | Logs: {out_log}", flush=True)
        time.sleep(2) # Stagger process creation to avoid ROCm allocation contention

    # Save PID manifest
    pid_file = os.path.join(TEMP_DIR, "parallel_pids.json")
    with open(pid_file, "w") as f:
        json.dump([{"run_id": p["run_id"], "pid": p["pid"], "out_log": p["out_log"], "log_json": p["log_json"]} for p in processes], f, indent=2)

    print("\nAll 6 processes successfully spawned concurrently on MI300X!")
    print(f"Tracking metadata saved to: {pid_file}")
    print("Beginning supervisory monitoring loop...\n", flush=True)

    ist_tz = timezone(timedelta(hours=5, minutes=30))

    while True:
        alive_count = sum(1 for p in processes if p["p"].poll() is None)
        current_utc = datetime.now(timezone.utc)
        current_ist = current_utc.astimezone(ist_tz)

        total_tokens = 0
        agg_tok_s = 0.0
        max_rem_s = 0.0

        print("\n" + "=" * 92)
        print(f" CARVE PARALLEL PRETRAINING MONITOR | Time: {current_ist.strftime('%Y-%m-%d %I:%M:%S %p IST')} | Active: {alive_count}/6")
        print("-" * 92)
        print(f"{'Run ID':<15} | {'PID':<7} | {'Step':<9} | {'Tokens Seen':<12} | {'Train Loss':<10} | {'Val Loss':<9} | {'Tok/s':<10} | {'Status'}")
        print("-" * 92)

        for p_info in processes:
            run_id = p_info["run_id"]
            pid = p_info["pid"]
            json_path = p_info["log_json"]
            poll_ret = p_info["p"].poll()
            status = "RUNNING" if poll_ret is None else ("COMPLETED" if poll_ret == 0 else f"FAILED({poll_ret})")

            step = 0
            total_steps = 4578
            tok_seen = 0
            train_loss = float("nan")
            val_loss = float("nan")
            tok_s = 0.0

            if os.path.exists(json_path):
                try:
                    with open(json_path, "r") as f:
                        h = json.load(f)
                    total_steps = h.get("total_steps", 4578)
                    if h.get("step"):
                        step = h["step"][-1]
                        tok_seen = h["tokens_seen"][-1]
                        train_loss = h["train_loss"][-1]
                        val_loss = h["val_loss"][-1] if h.get("val_loss") else float("nan")
                        tok_s = h["throughput_tokens_per_s"][-1] if h.get("throughput_tokens_per_s") else 0.0
                except Exception:
                    pass

            total_tokens += tok_seen
            agg_tok_s += tok_s
            rem_tok = max(0, 300_000_000 - tok_seen)
            rem_s = rem_tok / max(1.0, tok_s) if tok_s > 0 else 3600.0
            if rem_s > max_rem_s:
                max_rem_s = rem_s

            v_str = f"{val_loss:.4f}" if not math.isnan(val_loss) else "--"
            t_str = f"{train_loss:.4f}" if not math.isnan(train_loss) else "--"
            print(f"{run_id:<15} | {pid:<7} | {step:4d}/{total_steps} | {tok_seen/1e6:7.2f}M   | {t_str:<10} | {v_str:<9} | {tok_s:<10,.0f} | {status}")

        print("-" * 92)
        print(f"Total Tokens Processed Across 6 Runs : {total_tokens/1e6:,.2f} M / 1,800.00 M ({total_tokens/1.8e9*100:.1f}%)")
        print(f"Aggregate MI300X GPU Throughput       : {agg_tok_s:,.0f} tokens/s")
        eta_ist = current_ist + timedelta(seconds=max_rem_s)
        hrs = int(max_rem_s // 3600)
        mins = int((max_rem_s % 3600) // 60)
        secs = int(max_rem_s % 60)
        print(f"Estimated Time Remaining             : {hrs}h {mins}m {secs}s")
        print(f"Estimated Completion (IST)           : >>> {eta_ist.strftime('%Y-%m-%d %I:%M:%S %p IST')} <<<")
        print("=" * 92, flush=True)

        if alive_count == 0:
            print("\nAll 6 runs have completed!", flush=True)
            break

        time.sleep(15)

if __name__ == "__main__":
    main()
