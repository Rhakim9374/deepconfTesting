"""
Quick check: do warmup and final traces draw from the same min_conf distribution?

For each trace we compute (min_conf - conf_bar), where conf_bar is set per-question
from that question's warmup. Centering on conf_bar makes traces from different
questions comparable. We then aggregate across all (qid, rid) cells and overlay
warmup vs final.

If the two distributions overlap perfectly, the discrepancy in the analyzer
(~12% warmup pass vs ~28% final pass) is entirely a small-sample bias in
np.percentile(16, 90). If final is visibly shifted right, the final phase is
producing systematically higher min_conf than warmup.

Run on a machine where the pkl files live, e.g.:
  python examples/example_plot_conf_distributions.py \
      --output_dir /data/users1/rhakim/logs/deepconfTesting/qwen_gpqa_diamond \
      --max_qid 197 --rids 0 1 2 \
      --out qwen_conf_dist.png
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_qid", type=int, required=True)
    parser.add_argument("--rids", nargs="+", required=True)
    parser.add_argument("--out", default="conf_distributions.png")
    args = parser.parse_args()

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

    warmup_deltas, final_deltas = [], []
    final_completed_deltas = []
    n_warmup_above, n_final_above = 0, 0
    n_warmup, n_final = 0, 0

    for f in tqdm(keep):
        with open(f, "rb") as fh:
            r = pickle.load(fh)
        conf_bar = r.get("conf_bar")
        if conf_bar is None:
            continue
        for t in r.get("warmup_traces") or []:
            mc = t.get("min_conf")
            if mc is None:
                continue
            d = mc - conf_bar
            warmup_deltas.append(d)
            n_warmup += 1
            if d >= 0:
                n_warmup_above += 1
        for t in r.get("final_traces") or []:
            mc = t.get("min_conf")
            if mc is None:
                continue
            d = mc - conf_bar
            final_deltas.append(d)
            n_final += 1
            if d >= 0:
                n_final_above += 1
            if t.get("stop_reason") != "gconf_threshold":
                final_completed_deltas.append(d)

    warmup_deltas = np.asarray(warmup_deltas)
    final_deltas = np.asarray(final_deltas)
    final_completed_deltas = np.asarray(final_completed_deltas)

    lo = float(min(warmup_deltas.min(), final_deltas.min()))
    hi = float(max(warmup_deltas.max(), final_deltas.max()))
    bins = np.linspace(lo, hi, 80)

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    ax = axes[0]
    ax.hist(warmup_deltas, bins=bins, alpha=0.5, density=True,
            label=f"warmup (n={n_warmup})", color="C0")
    ax.hist(final_deltas, bins=bins, alpha=0.5, density=True,
            label=f"final all (n={n_final})", color="C1")
    ax.axvline(0, color="red", ls="--", lw=1, label="conf_bar (threshold)")
    ax.set_xlabel("min_conf - conf_bar  (per-question centered)")
    ax.set_ylabel("density")
    ax.set_title("Per-trace centered min_conf — warmup vs final")
    ax.legend()

    ax = axes[1]
    for arr, lbl, c in [
        (warmup_deltas, "warmup", "C0"),
        (final_deltas, "final all", "C1"),
        (final_completed_deltas, "final completed (post-relabel)", "C2"),
    ]:
        xs = np.sort(arr)
        ys = np.arange(1, len(xs) + 1) / len(xs)
        ax.plot(xs, ys, label=f"{lbl} (n={len(xs)})", color=c, lw=2)
    ax.axvline(0, color="red", ls="--", lw=1, label="conf_bar")
    ax.set_xlabel("min_conf - conf_bar")
    ax.set_ylabel("empirical CDF")
    ax.set_title("CDFs")
    ax.legend()

    plt.tight_layout()
    plt.savefig(args.out, dpi=120)

    print(f"warmup:  {n_warmup_above}/{n_warmup} above conf_bar "
          f"({100*n_warmup_above/n_warmup:.1f}%)")
    print(f"final:   {n_final_above}/{n_final} above conf_bar "
          f"({100*n_final_above/n_final:.1f}%)")
    print(f"warmup  median delta = {np.median(warmup_deltas):+.3f}")
    print(f"final   median delta = {np.median(final_deltas):+.3f}")
    print(f"warmup  mean   delta = {np.mean(warmup_deltas):+.3f}")
    print(f"final   mean   delta = {np.mean(final_deltas):+.3f}")
    print(f"Saved figure to {args.out}")


if __name__ == "__main__":
    main()
