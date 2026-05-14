
#!/usr/bin/env python3
import argparse, subprocess, sys, statistics, json, os
from pathlib import Path

def avg_from_jsonl(path: Path):
    vals = []
    with path.open("r", encoding="utf-8") as f:
        for ln in f:
            try:
                rec = json.loads(ln)
                if "avg" in rec:
                    vals.append(rec["avg"])
            except Exception:
                pass
    return statistics.mean(vals) if vals else 0.0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--provider", default="openai:gpt-4o-mini")
    ap.add_argument("--dataset", default="geo")
    ap.add_argument("--taus", default="0.3,0.4,0.5,0.6,0.7,0.8")
    ap.add_argument("--out", default="tau_sweep")
    args = ap.parse_args()

    data_root = Path(args.data_root)
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    best_tau, best_avg = None, -1.0
    for tau in [float(x) for x in args.taus.split(",")]:
        out_jsonl = out_dir / f"{args.dataset}_tau{tau}.jsonl"
        cmd = [
            sys.executable, "-m", "py_galois.runner",
            "--data-root", str(data_root),
            "--datasets", args.dataset,
            "--provider", args.provider,
            "--tau", str(tau),
            "--out", str(out_dir),
            "--mode", "galois_f"
        ]
        subprocess.run(cmd, check=True)
        avg = avg_from_jsonl(out_jsonl)
        print(f"tau={tau} -> AVG={avg:.3f}")
        if avg > best_avg:
            best_avg, best_tau = avg, tau
    print(f"Best tau={best_tau} (AVG={best_avg:.3f})")

if __name__ == "__main__":
    main()
