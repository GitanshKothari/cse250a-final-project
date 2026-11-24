from __future__ import annotations

import argparse
import subprocess
import time
from pathlib import Path


def mem_available_gb() -> float:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / 1e6
    return 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("jobs", type=Path)
    ap.add_argument("--max-procs", type=int, default=6)
    ap.add_argument("--min-free-gb", type=float, default=4.0)
    ap.add_argument("--poll", type=float, default=15.0)
    a = ap.parse_args()
    started: set[int] = set()
    running: dict[int, subprocess.Popen] = {}
    while True:
        lines = a.jobs.read_text().splitlines()
        for i, p in list(running.items()):
            if p.poll() is not None:
                print(f"{time.strftime('%H:%M:%S')} done  [{i}] exit {p.returncode}", flush=True)
                del running[i]
        pending = [i for i, ln in enumerate(lines)
                   if i not in started and ln.strip() and not ln.startswith("#")]
        if pending and len(running) < a.max_procs and mem_available_gb() >= a.min_free_gb:
            i = pending[0]
            log, cmd = lines[i].split("\t", 1)
            Path(log).parent.mkdir(parents=True, exist_ok=True)
            running[i] = subprocess.Popen(cmd, shell=True, stdout=open(log, "a"),
                                          stderr=subprocess.STDOUT, start_new_session=True)
            started.add(i)
            print(f"{time.strftime('%H:%M:%S')} start [{i}] (free {mem_available_gb():.1f} GB) "
                  f"{cmd[:110]}", flush=True)
            time.sleep(5)
            continue
        if not pending and not running and any(ln.strip() == "#END" for ln in lines):
            print("queue finished", flush=True)
            return
        time.sleep(a.poll)


if __name__ == "__main__":
    main()
