"""
Diagnostic zoom: where do `final_completed` traces actually sit relative to conf_bar?

The puzzle: ~28% of final traces have min_conf >= conf_bar, but the warmup
distribution only puts ~12% above. This script splits final traces by their
post-hoc bucket and bins the deltas, with a focus on whether final_completed
traces are crammed against the boundary (precision artifact) or spread out
(genuinely high-confidence).

Run on a machine where the pkl files live, e.g.:
  python examples/example_zoom_final_completed.py \\
      --output_dir /data/users1/rhakim/logs/deepconfTesting/qwen_gpqa_diamond \\
      --max_qid 197 --rids 0 1 2 \\
      --out qwen_final_zoom.png
"""
import argparse
import pickle
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


BUCKETS = [
    (-np.inf, -0.5, "(-inf, -0.5)"),
    (-0.5, -0.05, "[-0.5, -0.05)"),
    (-0.05, -0.01, "[-0.05, -0.01)"),
    (-0.01, -0.001, "[-0.01, -0.001)"),
    (-0.001, 0.0, "[-0.001, 0)"),
    (0.0, 0.001, "[0, 0.001)"),
    (0.001, 0.01, "[0.001, 0.01)"),
    (0.01, 0.05, "[0.01, 0.05)"),
    (0.05, 0.5, "[0.05, 0.5)"),
    (0.5, np.inf, "[0.5, +inf)"),
]


def bucket_breakdown(deltas, label):
    print(f"\n{label} (n={len(deltas)})")
    print("-" * 50)
    if len(deltas) == 0:
        print("  (empty)")
        return
    print(f"  min = {deltas.min():+.6f}")
    print(f"  max = {deltas.max():+.6f}")
    print(f"  median = {np.median(deltas):+.6f}")
    print(f"  mean = {deltas.mean():+.6f}")
    print()
    for lo, hi, name in BUCKETS:
        n = int(np.sum((deltas >= lo) & (deltas < hi)))
        pct = 100.0 * n / len(deltas)
        print(f"  {name:<22} {n:>6} ({pct:5.1f}%)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output_dir", required=True)
    p.add_argument("--max_qid", type=int, required=True)
    p.add_argument("--rids", nargs="+", required=True)
    p.add_argument("--out", default="final_zoom.png")
    args = p.parse_args()

    files = sorted(Path(args.output_dir).glob("deepthink_online_*.pkl"))
    qid_re = re.compile(r"deepthink_online_qid(\d+)_rid([^_]+)_")
    keep = []
    for f in files:
        m = qid_re.search(f.name)
        if not m:
            continue
        qid, rid = int(m.group(1)), m.group(2)
        if qid <= args.max_qid and rid in args.rids:
            keep.append(f)

    completed_deltas = []
    early_deltas = []
    conf_bar_values = []
    min_conf_values_completed = []
    n_total = 0
    for f in tqdm(keep):
        with open(f, "rb") as fh:
            r = pickle.load(fh)
        cb = r.get("conf_bar")
        if cb is None:
            continue
        conf_bar_values.append(cb)
        for t in r.get("final_traces") or []:
            mc = t.get("min_conf")
            if mc is None:
                continue
            n_total += 1
            d = mc - cb
            if t.get("stop_reason") == "gconf_threshold":
                early_deltas.append(d)
            else:
                completed_deltas.append(d)
                min_conf_values_completed.append(mc)

    completed = np.asarray(completed_deltas)
    early = np.asarray(early_deltas)
    conf_bars = np.asarray(conf_bar_values)
    completed_minconf = np.asarray(min_conf_values_completed)

    print(f"\nTotal final traces: {n_total}")
    print(f"  final_completed: {len(completed)} ({100*len(completed)/n_total:.1f}%)")
    print(f"  early_stopped:   {len(early)} ({100*len(early)/n_total:.1f}%)")

    bucket_breakdown(completed, "final_completed: min_conf - conf_bar")
    bucket_breakdown(early, "early_stopped: min_conf - conf_bar")

    # Boundary check: is conf_bar typically a 3-decimal value (likely tie)
    # vs an interpolated 4+-decimal value?
    cb_fractional = np.abs((conf_bars * 1000) - np.round(conf_bars * 1000))
    n_three_dec = int(np.sum(cb_fractional < 1e-6))
    print(f"\nconf_bar values (n={len(conf_bars)})")
    print("-" * 50)
    print(f"  conf_bar appears 3-decimal exact: {n_three_dec} ({100*n_three_dec/len(conf_bars):.1f}%)")
    print(f"  conf_bar appears interpolated:    {len(conf_bars)-n_three_dec} ({100*(len(conf_bars)-n_three_dec)/len(conf_bars):.1f}%)")
    print(f"  min = {conf_bars.min():.6f}")
    print(f"  max = {conf_bars.max():.6f}")
    print(f"  mean = {conf_bars.mean():.6f}")

    # ---- Plot ----
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    # (0,0): zoom near zero on final_completed
    ax = axes[0, 0]
    if len(completed) > 0:
        mask = (completed >= -0.001) & (completed <= 0.05)
        bins = np.linspace(-0.001, 0.05, 60)
        ax.hist(completed[mask], bins=bins, color="C2", alpha=0.8,
                label=f"final_completed (n in view: {int(mask.sum())})")
        ax.axvline(0, color="red", ls="--", lw=1, label="conf_bar")
        ax.set_xlabel("min_conf - conf_bar")
        ax.set_ylabel("count")
        ax.set_title("final_completed — extreme zoom [-0.001, 0.05]")
        ax.legend()

    # (0,1): medium zoom on final_completed
    ax = axes[0, 1]
    if len(completed) > 0:
        mask = completed <= 0.5
        bins = np.linspace(0, 0.5, 80)
        ax.hist(completed[mask], bins=bins, color="C2", alpha=0.8,
                label=f"final_completed (n in view: {int(mask.sum())})")
        ax.axvline(0, color="red", ls="--", lw=1, label="conf_bar")
        ax.set_xlabel("min_conf - conf_bar")
        ax.set_ylabel("count")
        ax.set_title("final_completed — medium zoom [0, 0.5]")
        ax.legend()

    # (1,0): full range of final_completed
    ax = axes[1, 0]
    if len(completed) > 0:
        ax.hist(completed, bins=80, color="C2", alpha=0.8,
                label=f"final_completed (n={len(completed)})")
        ax.axvline(0, color="red", ls="--", lw=1, label="conf_bar")
        ax.set_xlabel("min_conf - conf_bar")
        ax.set_ylabel("count")
        ax.set_title("final_completed — full range")
        ax.legend()

    # (1,1): early_stopped for comparison
    ax = axes[1, 1]
    if len(early) > 0:
        ax.hist(early, bins=80, color="C3", alpha=0.6,
                label=f"early_stopped (n={len(early)})")
        ax.axvline(0, color="red", ls="--", lw=1, label="conf_bar")
        ax.set_xlabel("min_conf - conf_bar")
        ax.set_ylabel("count")
        ax.set_title("early_stopped — full range")
        ax.legend()

    plt.tight_layout()
    plt.savefig(args.out, dpi=120)
    print(f"\nSaved figure to {args.out}")


if __name__ == "__main__":
    main()
