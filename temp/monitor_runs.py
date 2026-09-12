import os
import sys
import json
import time
import math
from datetime import datetime, timezone, timedelta

TEMP_DIR = "/root/carve/temp"
LOG_DIR = os.path.join(TEMP_DIR, "logs")
PID_FILE = os.path.join(TEMP_DIR, "parallel_pids.json")

def get_ist_time(utc_dt=None):
    if utc_dt is None:
        utc_dt = datetime.now(timezone.utc)
    ist_tz = timezone(timedelta(hours=5, minutes=30))
    return utc_dt.astimezone(ist_tz)

def format_ist(dt):
    return dt.strftime("%Y-%m-%d %I:%M:%S %p IST")

def main():
    if not os.path.exists(PID_FILE):
        print(f"No parallel runs found at {PID_FILE}")
        return

    with open(PID_FILE, "r") as f:
        procs = json.load(f)

    current_utc = datetime.now(timezone.utc)
    current_ist = get_ist_time(current_utc)

    print("================================================================================")
    print(f"       CARVE MULTI-RUN PARALLEL MONITOR: AMD INSTINCT MI300X                   ")
    print(f"       Current Time: {format_ist(current_ist)} ({current_utc.strftime('%H:%M:%S UTC')})       ")
    print("================================================================================")

    # Hardware status
    import subprocess
    smi_res = subprocess.run(["rocm-smi", "--showmeminfo", "vram", "--showuse", "--showpower"], capture_output=True, text=True)
    for line in smi_res.stdout.splitlines():
        if any(k in line for k in ["VRAM Total", "VRAM Used", "GPU Use", "Average Power", "Power"]):
            print(f" [HW] {line.strip()}")
    print("-" * 88)

    total_tokens_seen = 0
    aggregate_throughput = 0.0
    runs_status = []
    max_remaining_s = 0.0

    print(f"{'Run ID':<15} | {'PID':<7} | {'Step':<9} | {'Tokens Seen':<12} | {'Train Loss':<10} | {'Val Loss':<9} | {'Tok/s':<10} | {'State'}")
    print("-" * 88)

    for p_info in procs:
        run_id = p_info["run_id"]
        pid = p_info["pid"]
        json_path = p_info["log_json"]
        out_log = p_info["out_log"]

        # Check process alive
        is_alive = False
        try:
            os.kill(pid, 0)
            is_alive = True
        except OSError:
            is_alive = False

        status_state = "RUNNING" if is_alive else "EXITED"

        step = 0
        total_steps = 4578
        tok_seen = 0
        train_loss = float("nan")
        val_loss = float("nan")
        tok_s = 0.0
        elapsed_s = 0.0

        if os.path.exists(json_path):
            try:
                with open(json_path, "r") as f:
                    h = json.load(f)
                total_steps = h.get("total_steps", 4578)
                if h.get("step"):
                    step = h["step"][-1]
                    tok_seen = h["tokens_seen"][-1]
                    train_loss = h["train_loss"][-1]
                    val_loss = h["val_loss"][-1] if h["val_loss"] else float("nan")
                    tok_s = h["throughput_tokens_per_s"][-1] if h["throughput_tokens_per_s"] else 0.0
                    elapsed_s = h["wallclock_elapsed_s"][-1] if h["wallclock_elapsed_s"] else 0.0
            except Exception:
                pass

        if step == 0 and os.path.exists(out_log):
            # Parse from stdout if json not yet flushed
            with open(out_log, "r") as f:
                lines = f.readlines()[-10:]
            for l in reversed(lines):
                if "Step" in l and "tok/s" in l:
                    parts = l.strip().split("|")
                    try:
                        step_part = parts[0].split("Step")[1].split()[0]
                        step = int(step_part.split("/")[0])
                        tok_s = float(l.split("tok/s")[0].split("|")[-1].replace(",", "").strip())
                    except Exception:
                        pass
                    break

        total_tokens_seen += tok_seen
        aggregate_throughput += tok_s

        # Remaining time for this run
        remaining_tokens = max(0, 300_000_000 - tok_seen)
        if tok_s > 0:
            rem_s = remaining_tokens / tok_s
        else:
            rem_s = remaining_tokens / 100_000 # default estimate

        if rem_s > max_remaining_s:
            max_remaining_s = rem_s

        val_str = f"{val_loss:.4f}" if not math.isnan(val_loss) else "--"
        t_loss_str = f"{train_loss:.4f}" if not math.isnan(train_loss) else "--"
        step_str = f"{step}/{total_steps}"

        print(f"{run_id:<15} | {pid:<7} | {step_str:<9} | {tok_seen/1e6:7.2f}M   | {t_loss_str:<10} | {val_str:<9} | {tok_s:<10,.0f} | {status_state}")

    print("=" * 88)
    print(f"Total Tokens Processed across 6 Runs : {total_tokens_seen/1e6:,.2f} M / 1,800.00 M ({total_tokens_seen/1.8e9*100:.1f}%)")
    print(f"Aggregate MI300X GPU Throughput       : {aggregate_throughput:,.0f} tokens / second")

    # Calculate ETA in IST
    eta_seconds = max_remaining_s
    eta_utc = current_utc + timedelta(seconds=eta_seconds)
    eta_ist = get_ist_time(eta_utc)

    hrs = int(eta_seconds // 3600)
    mins = int((eta_seconds % 3600) // 60)
    secs = int(eta_seconds % 60)

    print(f"Estimated Remaining Wallclock Time   : {hrs}h {mins}m {secs}s ({eta_seconds:,.0f} seconds)")
    print(f"Estimated Completion Time (IST)      : >>> {format_ist(eta_ist)} <<<")
    print("================================================================================")

if __name__ == "__main__":
    main()
