"""Convert raw GPQA Diamond CSV into the project's JSONL eval format.

Each output row:
  {
    "question":   <question>\\n\\n<a..d options with text>\\n\\n<instruction>,
    "answer":     "A" | "B" | "C" | "D",
    "unique_id":  <Record ID from source>,
    "subdomain":  <Subdomain from source>
  }

Single labeling scheme. The four options are randomly shuffled into
lowercase slots a, b, c, d (seeded from Record ID for reproducibility), and
the gold answer is the uppercase letter naming the slot that holds the
correct option. Grading is case-insensitive on single-letter answers (see
example_online.py::equal_func), so the model can respond with `c` or `C`
or `\\boxed{C}` or `\\boxed{\\text{c}}` and all four match.

One CSV format quirk: the source GPQA Diamond CSV occasionally stores the
options inline in the Question text and uses single letters (`a..d`) as
Correct Answer / Incorrect Answer N. Those rows are detected and passed
through without a re-shuffle: the inline options become the lowercase
block and the gold uppercase letter is just the inline letter, upcased.

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


def _seed(record_id: str) -> int:
    return int(hashlib.sha256(record_id.encode("utf-8")).hexdigest(), 16) % (2**32)


def _is_inline_letter(text: str) -> bool:
    """True if `text` is a single a-d letter (Format B row indicator)."""
    return len(text) == 1 and text.lower() in "abcd"


def build_row(row: dict) -> dict:
    correct = row["Correct Answer"].strip()
    incorrect = [row[f"Incorrect Answer {i}"].strip() for i in (1, 2, 3)]
    question_text = row["Question"].strip()

    if _is_inline_letter(correct) and all(_is_inline_letter(x) for x in incorrect):
        # Format B: question already contains the lowercase a..d options inline.
        # No new shuffle — the source CSV's "Correct Answer" letter is the gold.
        question = "\n\n".join([question_text, INSTRUCTION])
        answer = correct.upper()
    else:
        # Format A: build a fresh lowercase block from the four option texts,
        # using a per-row seeded shuffle so the build is reproducible.
        options = [correct] + incorrect  # index 0 is the correct option
        rng = random.Random(_seed(row["Record ID"]))
        lower_perm = list(range(4))
        rng.shuffle(lower_perm)
        lower_text = [options[lower_perm[i]] for i in range(4)]
        correct_lower = lower_perm.index(0)

        lower_block = "\n".join(f"{LOWER[i]}) {lower_text[i]}" for i in range(4))
        question = "\n\n".join([question_text, lower_block, INSTRUCTION])
        answer = LOWER[correct_lower].upper()

    return {
        "question": question,
        "answer": answer,
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
