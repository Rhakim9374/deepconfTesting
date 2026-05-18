"""Convert raw GPQA Diamond CSV into the project's JSONL eval format.

Each output row:
  {
    "question":   <question>\n\n<a..d options>\n\n<A..D -> a..d mapping>\n\n<instruction>,
    "answer":     "A" | "B" | "C" | "D",
    "unique_id":  <Record ID from source>,
    "subdomain":  <Subdomain from source>
  }

Two independent shuffles per row, both seeded from Record ID for reproducibility:
  1. The four options are placed into lowercase slots a, b, c, d in a random order.
  2. A separate random permutation maps each uppercase letter A..D to one lowercase slot.
The gold answer is the uppercase letter whose mapped lowercase slot holds the correct option.

Usage:
  python scripts/build_gpqa_jsonl.py \\
      --csv data/gpqa_diamond/gpqa_diamond.csv \\
      --out data/gpqa_diamond.jsonl
"""

import argparse
import csv
import hashlib
import json
import random
from pathlib import Path

INSTRUCTION = (
    "Your final answer should only contain the letter corresponding to the correct choice."
)
LOWER = ["a", "b", "c", "d"]
UPPER = ["A", "B", "C", "D"]


def _seed(record_id: str) -> int:
    return int(hashlib.sha256(record_id.encode("utf-8")).hexdigest(), 16) % (2**32)


def build_row(row: dict) -> dict:
    correct = row["Correct Answer"].strip()
    incorrect = [row[f"Incorrect Answer {i}"].strip() for i in (1, 2, 3)]
    options = [correct] + incorrect  # index 0 is the correct option

    rng = random.Random(_seed(row["Record ID"]))

    # Shuffle the four options into lowercase positions a..d.
    lower_perm = list(range(4))
    rng.shuffle(lower_perm)
    lower_text = [options[lower_perm[i]] for i in range(4)]
    correct_lower = lower_perm.index(0)  # which lowercase slot holds the correct option

    # Independent shuffle for the uppercase -> lowercase mapping.
    upper_to_lower = list(range(4))
    rng.shuffle(upper_to_lower)
    correct_upper = upper_to_lower.index(correct_lower)

    lower_block = "\n".join(f"{LOWER[i]}) {lower_text[i]}" for i in range(4))
    upper_block = "\n".join(f"{UPPER[i]}. {LOWER[upper_to_lower[i]]}" for i in range(4))
    question = "\n\n".join([row["Question"].strip(), lower_block, upper_block, INSTRUCTION])

    return {
        "question": question,
        "answer": UPPER[correct_upper],
        "unique_id": row["Record ID"].strip(),
        "subdomain": row.get("Subdomain", "").strip(),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv", required=True, type=Path, help="Path to raw gpqa_diamond.csv")
    p.add_argument("--out", required=True, type=Path, help="Output JSONL path")
    args = p.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with args.csv.open(newline="") as f, args.out.open("w") as out:
        reader = csv.DictReader(f)
        for row in reader:
            if not row.get("Question") or not row.get("Correct Answer"):
                continue
            out.write(json.dumps(build_row(row), ensure_ascii=False) + "\n")
            n += 1
    print(f"Wrote {n} rows to {args.out}")


if __name__ == "__main__":
    main()
