#!/usr/bin/env python3
"""
Full-dataset online runner for DeepConf.

Loads the vLLM model once, then loops over all questions in the dataset,
running online-mode deep thinking on each. Designed to be called via
HTCondor with `queue NUM_RUNS` (one job per run, each processing all questions).
"""
import argparse
import json
import os
import pickle
import time
from datetime import datetime

from vllm import SamplingParams

from .wrapper import DeepThinkLLM
from examples.example_online import (
    prepare_prompt,
    prepare_prompt_gpt,
    equal_func,
    evaluate_voting_results,
    evaluate_confidence_methods,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run DeepConf online mode on an entire dataset."
    )
    parser.add_argument("--model", type=str, required=True, help="Model path or name")
    parser.add_argument(
        "--dataset", type=str, required=True, help="Path to JSONL dataset"
    )
    parser.add_argument(
        "--run_id", type=str, default="0", help="Run ID for distinguishing repeats"
    )
    parser.add_argument("--warmup_traces", type=int, default=16)
    parser.add_argument("--total_budget", type=int, default=256)
    parser.add_argument("--confidence_percentile", type=int, default=10)
    parser.add_argument("--window_size", type=int, default=2048)
    parser.add_argument("--max_tokens", type=int, default=64000)
    parser.add_argument(
        "--model_type",
        type=str,
        default="deepseek",
        choices=["deepseek", "gpt"],
    )
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument(
        "--no_multiple_voting",
        action="store_true",
        help="Disable multiple voting analysis",
    )
    parser.add_argument(
        "--qid_start", type=int, default=None, help="First question index (inclusive)"
    )
    parser.add_argument(
        "--qid_end", type=int, default=None, help="Last question index (exclusive)"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # ------------------------------------------------------------------
    # Load dataset
    # ------------------------------------------------------------------
    print(f"Loading dataset from {args.dataset}...")
    with open(args.dataset, "r", encoding="utf-8") as f:
        data = [json.loads(line.strip()) for line in f if line.strip()]
    print(f"Dataset contains {len(data)} questions.")

    qid_start = args.qid_start if args.qid_start is not None else 0
    qid_end = args.qid_end if args.qid_end is not None else len(data)
    qid_end = min(qid_end, len(data))
    print(f"Processing questions [{qid_start}, {qid_end})")

    # ------------------------------------------------------------------
    # Load model once
    # ------------------------------------------------------------------
    print(f"Loading model {args.model} ...")
    t0 = time.time()
    deep_llm = DeepThinkLLM(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        enable_prefix_caching=True,
    )
    print(f"Model loaded in {time.time() - t0:.1f}s")

    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_tokens=args.max_tokens,
        logprobs=20, # back to 20 default
    )

    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # ------------------------------------------------------------------
    # Per-question loop
    # ------------------------------------------------------------------
    summary_results = []

    for qid in range(qid_start, qid_end):
        question_data = data[qid]
        question = question_data["question"]
        ground_truth = str(question_data.get("answer", "")).strip()

        print(f"\n{'='*80}")
        print(f"[qid={qid}] {question[:120]}...")
        print(f"Ground truth: {ground_truth}")
        print(f"{'='*80}")

        # Prepare prompt (same logic as example_online.py)
        if args.model_type == "gpt":
            prompt = prepare_prompt_gpt(question, deep_llm.tokenizer)
        else:
            prompt = prepare_prompt(question, deep_llm.tokenizer, args.model_type)

        # Run online deep thinking
        q_start = time.time()
        result = deep_llm.deepthink(
            prompt=prompt,
            mode="online",
            warmup_traces=args.warmup_traces,
            total_budget=args.total_budget,
            confidence_percentile=args.confidence_percentile,
            window_size=args.window_size,
            compute_multiple_voting=not args.no_multiple_voting,
            sampling_params=sampling_params,
        )
        q_elapsed = time.time() - q_start

        # Evaluate
        evaluation = None
        confidence_eval = None
        if ground_truth:
            if result.voting_results:
                evaluation = evaluate_voting_results(result.voting_results, ground_truth)
            confidence_eval = evaluate_confidence_methods(result, ground_truth)

        # Print short status
        if evaluation:
            correct_methods = [m for m, e in evaluation.items() if e["is_correct"]]
            print(f"  Correct methods: {correct_methods}")
        print(f"  Time: {q_elapsed:.1f}s | Tokens: {getattr(result, 'total_tokens', '?')}")

        # Save per-question pkl (same format as example_online.py)
        result_data = result.to_dict()
        result_data.update(
            {
                "question": question,
                "ground_truth": ground_truth,
                "qid": qid,
                "run_id": args.run_id,
                "evaluation": evaluation,
                "confidence_evaluation": confidence_eval,
            }
        )

        pkl_path = os.path.join(
            args.output_dir,
            f"deepthink_online_qid{qid}_rid{args.run_id}_{timestamp}.pkl",
        )
        with open(pkl_path, "wb") as pf:
            pickle.dump(result_data, pf)
        print(f"  Saved: {pkl_path}")

        # Collect for summary
        summary_results.append(
            {
                "qid": qid,
                "question": question,
                "ground_truth": ground_truth,
                "evaluation": evaluation,
                "confidence_evaluation": confidence_eval,
                "total_tokens": getattr(result, "total_tokens", None),
                "warmup_tokens": getattr(result, "warmup_tokens", None),
                "final_tokens": getattr(result, "final_tokens", None),
                "total_time": getattr(result, "total_time", None),
                "q_elapsed": q_elapsed,
                "conf_bar": getattr(result, "conf_bar", None),
                "total_traces_count": getattr(result, "total_traces_count", None),
                "pkl_path": pkl_path,
            }
        )

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print(f"\n{'='*80}")
    print("AGGREGATE SUMMARY")
    print(f"{'='*80}")

    # Compute per-method accuracy across questions
    method_correct = {}
    method_total = {}
    for entry in summary_results:
        if entry["evaluation"]:
            for method, eval_result in entry["evaluation"].items():
                method_total[method] = method_total.get(method, 0) + 1
                if eval_result["is_correct"]:
                    method_correct[method] = method_correct.get(method, 0) + 1

    n_questions = len(summary_results)
    print(f"Questions processed: {n_questions}")
    print(f"Run ID: {args.run_id}")
    print()
    if method_total:
        print(f"{'Method':<30} {'Correct':<10} {'Total':<10} {'Accuracy':<10}")
        print("-" * 60)
        for method in sorted(method_total.keys()):
            correct = method_correct.get(method, 0)
            total = method_total[method]
            acc = correct / total if total > 0 else 0.0
            print(f"{method:<30} {correct:<10} {total:<10} {acc:<10.1%}")

    # Total tokens / time
    total_tokens = sum(e["total_tokens"] for e in summary_results if e["total_tokens"])
    total_time = sum(e["q_elapsed"] for e in summary_results)
    print(f"\nTotal tokens: {total_tokens}")
    print(f"Total wall time: {total_time:.1f}s")

    # Save summary pkl
    summary = {
        "run_id": args.run_id,
        "model": args.model,
        "dataset": args.dataset,
        "qid_start": qid_start,
        "qid_end": qid_end,
        "n_questions": n_questions,
        "timestamp": timestamp,
        "args": vars(args),
        "per_question": summary_results,
        "method_accuracy": {
            method: {
                "correct": method_correct.get(method, 0),
                "total": method_total[method],
                "accuracy": method_correct.get(method, 0) / method_total[method]
                if method_total[method] > 0
                else 0.0,
            }
            for method in sorted(method_total.keys())
        },
        "total_tokens": total_tokens,
        "total_time": total_time,
    }

    summary_path = os.path.join(
        args.output_dir, f"summary_online_rid{args.run_id}_{timestamp}.pkl"
    )
    with open(summary_path, "wb") as sf:
        pickle.dump(summary, sf)
    print(f"\nSummary saved to: {summary_path}")


if __name__ == "__main__":
    main()
