#!/usr/bin/env python3
import argparse
import json
import os
from datetime import datetime
from typing import Any, Dict, Optional

from transformers import AutoTokenizer
from vllm import SamplingParams

from .wrapper import DeepThinkLLM


def str2bool(v: str) -> bool:
    return str(v).strip().lower() in {"1", "true", "yes", "y", "on"}


def read_jsonl_row(path: str, index: int) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    if not rows:
        raise ValueError(f"Dataset is empty: {path}")
    if index < 0 or index >= len(rows):
        raise IndexError(f"QUESTION_INDEX={index} is out of range for dataset of size {len(rows)}")
    return rows[index]


def infer_prompt_and_answer(record: Dict[str, Any]) -> tuple[str, Optional[str]]:
    prompt = (
        record.get("question")
        or record.get("prompt")
        or record.get("problem")
        or record.get("input")
        or record.get("query")
    )
    if not prompt:
        raise KeyError(
            "Could not infer prompt field from dataset row. Expected one of: question, prompt, problem, input, query"
        )

    answer = record.get("answer")
    if answer is None:
        answer = record.get("target")
    if answer is None:
        answer = record.get("ground_truth")
    if answer is not None:
        answer = str(answer)
    return str(prompt), answer


def build_chat_prompt(model_path: str, question: str, trust_remote_code: bool) -> str:
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=trust_remote_code)
    messages = [{"role": "user", "content": question}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def output_to_dict(result: Any) -> Dict[str, Any]:
    data: Dict[str, Any] = {
        "final_answer": getattr(result, "final_answer", None),
        "voted_answer": getattr(result, "voted_answer", None),
        "conf_bar": getattr(result, "conf_bar", None),
        "total_traces_count": getattr(result, "total_traces_count", None),
        "config": getattr(result, "config", None),
        "voting_results": getattr(result, "voting_results", None),
    }
    for attr in ["warmup_traces", "final_traces", "all_traces"]:
        value = getattr(result, attr, None)
        if value is not None:
            data[attr] = value
    return data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run DeepConf from the repository-local deepconf/main.py module.")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--mode", choices=["offline", "online"], default="offline")
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--dataset-path", default=os.environ.get("DATASET_PATH"))
    parser.add_argument("--question-index", type=int, default=0)
    parser.add_argument("--budget", type=int, default=4)
    parser.add_argument("--warmup-traces", type=int, default=4)
    parser.add_argument("--total-budget", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--trust-remote-code", default="1")
    parser.add_argument("--compute-multiple-voting", default="1")
    parser.add_argument("--output-path", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    trust_remote_code = str2bool(args.trust_remote_code)
    compute_multiple_voting = str2bool(args.compute_multiple_voting)

    ground_truth = None
    if args.prompt:
        question = args.prompt
    elif args.dataset_path:
        row = read_jsonl_row(args.dataset_path, args.question_index)
        question, ground_truth = infer_prompt_and_answer(row)
    else:
        question = "What is the square root of 144?"

    prompt = build_chat_prompt(args.model_path, question, trust_remote_code)

    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_tokens=args.max_tokens,
    )

    deep_llm = DeepThinkLLM(
        model=args.model_path,
        tensor_parallel_size=args.tensor_parallel_size,
        trust_remote_code=trust_remote_code,
        enable_prefix_caching=True,
    )

    if args.mode == "offline":
        result = deep_llm.deepthink(
            prompt=prompt,
            mode="offline",
            budget=args.budget,
            compute_multiple_voting=compute_multiple_voting,
            sampling_params=sampling_params,
        )
    else:
        result = deep_llm.deepthink(
            prompt=prompt,
            mode="online",
            warmup_traces=args.warmup_traces,
            total_budget=args.total_budget,
            sampling_params=sampling_params,
        )

    payload = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "model_path": args.model_path,
        "mode": args.mode,
        "question_index": args.question_index,
        "question": question,
        "ground_truth": ground_truth,
        "result": output_to_dict(result),
    }

    print("=" * 80)
    print("DeepConf run complete")
    print(f"Mode:         {args.mode}")
    print(f"Model path:   {args.model_path}")
    if args.dataset_path:
        print(f"Dataset path: {args.dataset_path}")
        print(f"Question idx: {args.question_index}")
    print(f"Question:     {question}")
    print(f"Final answer: {payload['result'].get('final_answer')}")
    print(f"Voted answer: {payload['result'].get('voted_answer')}")
    if ground_truth is not None:
        print(f"Ground truth: {ground_truth}")
    if payload['result'].get('conf_bar') is not None:
        print(f"Conf bar:     {payload['result'].get('conf_bar')}")
    print("=" * 80)

    output_path = args.output_path
    if not output_path:
        safe_mode = args.mode.replace(os.sep, "_")
        output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "outputs")
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(
            output_dir,
            f"deepconf_{safe_mode}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        )

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"Saved output to: {output_path}")


if __name__ == "__main__":
    main()

