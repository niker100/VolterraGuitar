"""Stale-run detector for background GPU experiments.

Background tasks notify on normal EXIT, but a reaped (killed by an interrupt) or
hung job sends no notification — so a run can silently stop. This scans
outputs/logs/*.log and flags any whose last write was too long ago (the robust
wall-clock signal: a live training loop appends a timestamped line every ~30-70s),
cross-checked against GPU utilisation and live python processes.

Exit code 1 if any log is STALE (idle > --max-idle), else 0 — so the loop can
branch on it.

Run: uv run python check_runs.py [--max-idle 200] [--active-only]
"""

from __future__ import annotations

import argparse
import contextlib
import glob
import os
import subprocess
import sys
import time


def gpu_util() -> str:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=15,
        )
        return out.stdout.strip() or "n/a"
    except Exception:
        return "n/a"


def n_python() -> int:
    """Count python.exe processes whose command line looks like a training run."""
    try:
        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "(Get-CimInstance Win32_Process -Filter \"name='python.exe'\" | "
             "Where-Object { $_.CommandLine -match 'radical_|compare|benchmark|\\.py' }).Count"],
            capture_output=True, text=True, timeout=20,
        )
        return int((out.stdout or "0").strip() or 0)
    except Exception:
        return -1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-idle", type=float, default=200.0,
                    help="seconds of log silence before a run is STALE")
    ap.add_argument("--active-only", action="store_true",
                    help="only show logs touched in the last hour")
    args = ap.parse_args()

    logs = sorted(glob.glob("outputs/logs/*.log"), key=os.path.getmtime, reverse=True)
    now = time.time()
    gpu = gpu_util()
    util = 0.0
    with contextlib.suppress(Exception):
        util = float(gpu.split("%")[0])
    nproc = n_python()
    print(f"GPU: {gpu}   training-python procs: {nproc}")
    print(f"{'log':42s} {'idle':>8s}  state   last line")
    any_stale = False
    for f in logs:
        age = now - os.path.getmtime(f)
        if args.active_only and age > 3600:
            continue
        try:
            last = ""
            with open(f, encoding="utf-8", errors="replace") as fh:
                for line in fh:
                    if line.strip():
                        last = line.strip()
        except Exception:
            last = "<unreadable>"
        done = "VERDICT" in last or "wrote " in last or "===" in last
        if done:
            state = "done "
        elif age > args.max_idle:
            # stale: log silent too long. If GPU idle too, it's almost surely dead.
            state = "STALE!" if util < 5 else "stale?"
            any_stale = True
        else:
            state = "live "
        print(f"{os.path.basename(f):42s} {age:7.0f}s  {state}  {last[:80]}")
    if any_stale:
        print("\n!! STALE run(s) detected — check process/GPU and relaunch if dead.")
    return 1 if any_stale else 0


if __name__ == "__main__":
    sys.exit(main())
