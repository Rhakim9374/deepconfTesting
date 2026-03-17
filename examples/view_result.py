#!/usr/bin/env python3
"""View a DeepConf result pkl file with a clean, readable summary."""

import argparse
import pickle
import sys


def truncate(s, max_len=200):
    s = str(s).replace("\n", " ")
    return s[:max_len] + "..." if len(s) > max_len else s


def fmt_val(v):
    """Format a value for the --keys view."""
    t = type(v).__name__
    if isinstance(v, (list, tuple)):
        return f"{t}[{len(v)}]"
    if isinstance(v, dict):
        return f"{t}{{{len(v)} keys}}"
    if isinstance(v, str):
        return f"{t}({len(v)} chars)"
    return t


def print_keys(data):
    print("\n=== Top-level keys ===")
    for k in sorted(data.keys()):
        print(f"  {k:30s}  {fmt_val(data[k])}")


def print_evaluation(data):
    evaluation = data.get("evaluation", {})
    if not evaluation:
        return

    print("\n=== Evaluation ===")
    header = f"  {'Method':<35s} {'Answer':>10s} {'Correct':>8s} {'Conf':>8s} {'Votes':>6s}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for method, info in evaluation.items():
        answer = str(info.get("answer", ""))[:10]
        correct = "Yes" if info.get("is_correct") else "No"
        conf = info.get("confidence")
        conf_str = f"{conf:.4f}" if conf is not None else "-"
        votes = info.get("num_votes", "-")
        print(f"  {method:<35s} {answer:>10s} {correct:>8s} {conf_str:>8s} {str(votes):>6s}")


def print_confidence_evaluation(data):
    ce = data.get("confidence_evaluation")
    if not ce:
        return

    print("\n=== Confidence Evaluation ===")
    for category, info in ce.items():
        total = info.get("total", 0)
        correct = info.get("correct", 0)
        acc = info.get("accuracy", 0)
        print(f"  {category:<25s}  total={total}  correct={correct}  accuracy={acc:.4f}")


def print_traces(data):
    all_traces = data.get("all_traces", [])
    if not all_traces:
        print("\n(no traces found)")
        return

    print(f"\n=== Traces ({len(all_traces)}) ===")
    header = f"  {'#':>3s}  {'Answer':>10s}  {'Tokens':>7s}  {'MinConf':>8s}  {'StopReason'}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for i, t in enumerate(all_traces):
        answer = str(t.get("extracted_answer", ""))[:10]
        tokens = t.get("num_tokens", "?")
        min_conf = t.get("min_conf")
        if min_conf is None:
            # compute from confs if available
            confs = t.get("confs", [])
            min_conf = min(confs) if confs else None
        min_conf_str = f"{min_conf:.4f}" if min_conf is not None else "-"
        stop = t.get("stop_reason", "?")
        print(f"  {i:>3d}  {answer:>10s}  {tokens:>7}  {min_conf_str:>8s}  {stop}")


def print_summary(data):
    print("=== Result Summary ===")
    print(f"  Question:     {truncate(data.get('question', '?'))}")
    print(f"  Ground truth: {data.get('ground_truth', '?')}")
    print(f"  Mode:         {data.get('mode', '?')}")

    # Token stats
    ts = data.get("token_stats", {})
    if ts:
        print("\n=== Token Stats ===")
        for k in ["total_tokens", "warmup_tokens", "final_tokens",
                   "avg_tokens_per_trace", "avg_tokens_per_warmup_trace",
                   "avg_tokens_per_final_trace"]:
            v = ts.get(k)
            if v is not None:
                if isinstance(v, float):
                    print(f"  {k:<35s}  {v:.1f}")
                else:
                    print(f"  {k:<35s}  {v}")

    # Timing stats
    timing = data.get("timing_stats", {})
    if timing:
        print("\n=== Timing Stats ===")
        for k in ["total_time", "generation_time", "processing_time",
                   "warmup_gen_time", "final_gen_time"]:
            v = timing.get(k)
            if v is not None:
                print(f"  {k:<35s}  {v:.2f}s")

    # Conf bar (online mode)
    conf_bar = data.get("conf_bar")
    if conf_bar is not None:
        print(f"\n  Confidence bar: {conf_bar:.4f}")

    print_evaluation(data)
    print_confidence_evaluation(data)


def main():
    parser = argparse.ArgumentParser(description="View a DeepConf result pkl file.")
    parser.add_argument("pkl_file", help="Path to the pkl file")
    parser.add_argument("--traces", action="store_true",
                        help="Show per-trace summary table")
    parser.add_argument("--keys", action="store_true",
                        help="List top-level keys and their types/sizes")
    args = parser.parse_args()

    try:
        with open(args.pkl_file, "rb") as f:
            data = pickle.load(f)
    except FileNotFoundError:
        print(f"Error: file not found: {args.pkl_file}", file=sys.stderr)
        sys.exit(1)

    if args.keys:
        print_keys(data)
        return

    print_summary(data)

    if args.traces:
        print_traces(data)


if __name__ == "__main__":
    main()
