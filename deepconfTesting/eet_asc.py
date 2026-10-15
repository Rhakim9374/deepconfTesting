"""The ExploreExploitThink recorded-only Adaptive-Consistency replay,
mirrored here (``src/eet/asc.py``).

The DeepConf-online baseline logs when each of its native voters became
eligible (warm-up traces after calibration, final traces when they finish
unkilled) and :func:`shadow_record` replays the published Beta rule over
that log after the question has run to completion. Nothing stops early.
Every function is a copy of the ExploreExploitThink original; keep them in
sync (``tests/test_deepconf_voting_parity.py`` in that repository checks).
"""
import math
from collections import Counter
from fractions import Fraction
from typing import Dict, List, Optional, Sequence

from .eet_voting import canonical_answer, zsoftmax_vote

# The published threshold ``prob >= 0.95`` as an exact ratio.
ASC_THRESHOLD = Fraction(19, 20)


def beta_stop_probability(v1: int, v2: int) -> Fraction:
    """``P(X > 1/2)`` for ``X ~ Beta(v1 + 1, v2 + 1)``, exactly."""
    if v2 < 0 or v1 < v2:
        raise ValueError(f"need v1 >= v2 >= 0, got ({v1}, {v2})")
    n = v1 + v2 + 1
    return Fraction(sum(math.comb(n, j) for j in range(v1 + 1)), 2 ** n)


def asc_fires(counts: Dict[str, int]) -> bool:
    """The published stop test on canonical answer counts; no answers never fire."""
    top = sorted(counts.values(), reverse=True)
    if not top:
        return False
    v2 = top[1] if len(top) > 1 else 0
    return beta_stop_probability(top[0], v2) >= ASC_THRESHOLD


def pool_votes(pool: Sequence[int], answers: Sequence[Optional[str]],
               tails: Sequence[Optional[float]]) -> dict:
    """Majority and z-softmax answers of the traces ``pool`` in native order."""
    order = sorted(pool)
    counts: Counter = Counter()
    first_raw: Dict[str, str] = {}
    for i in order:
        if answers[i] is not None:
            key = canonical_answer(answers[i])
            counts[key] += 1
            first_raw.setdefault(key, answers[i])
    top = sorted(counts.values(), reverse=True)
    v1 = top[0] if top else 0
    v2 = top[1] if len(top) > 1 else 0
    return {
        "n_admitted": len(order),
        "n_answered": sum(counts.values()),
        "counts": dict(counts),
        "probability": float(beta_stop_probability(v1, v2)) if top else None,
        "majority_answer": first_raw[max(counts, key=counts.get)] if counts else None,
        "zsoftmax_answer": zsoftmax_vote(
            [answers[i] for i in order], [tails[i] for i in order])[0],
    }


def shadow_record(admissions: List[dict], checkpoints: List[dict],
                  answers: Sequence[Optional[str]],
                  tails: Sequence[Optional[float]],
                  full_time_s: float, full_tokens: int) -> dict:
    """Replay ASC over a run's admission log (copy of ``eet.asc.shadow_record``)."""
    def stop(stamp: dict, pool: List[int]) -> dict:
        return {"fired": True, "time_s": stamp["time_s"], "tokens": stamp["tokens"],
                "round": stamp["round"], "pool": list(pool), **pool_votes(pool, answers, tails)}

    order = sorted(range(len(admissions)),
                   key=lambda k: (admissions[k]["time_s"], admissions[k]["index"]))
    counts: Counter = Counter()
    pool: List[int] = []
    event = None
    for k in order:
        a = admissions[k]
        pool.append(a["index"])
        if answers[a["index"]] is not None:
            counts[canonical_answer(answers[a["index"]])] += 1
            if event is None and asc_fires(counts):
                event = stop(a, pool)
    checkpoint = None
    for c in checkpoints:
        members = [a["index"] for a in admissions[:c["n_admitted"]]]
        answered = Counter(canonical_answer(answers[i]) for i in members
                           if answers[i] is not None)
        if asc_fires(answered):
            checkpoint = stop(c, members)
            break
    full = {"fired": False, "time_s": full_time_s, "tokens": full_tokens,
            "round": checkpoints[-1]["round"] if checkpoints else 0,
            "pool": [a["index"] for a in admissions],
            **pool_votes([a["index"] for a in admissions], answers, tails)}
    return {
        "threshold": f"{ASC_THRESHOLD.numerator}/{ASC_THRESHOLD.denominator}",
        "event": event, "checkpoint": checkpoint, "full": full,
        "admissions": [dict(a, answered=answers[a["index"]] is not None)
                       for a in admissions],
        "checkpoints": list(checkpoints),
    }
