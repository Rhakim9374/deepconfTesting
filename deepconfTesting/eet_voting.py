"""The three ExploreExploitThink voting read-outs, mirrored here.

DeepConf's own voting (``utils.compute_all_voting_results``) is left
untouched. This module adds the three strategies the thesis reports for
every method, so a DeepConf run and an EET run are read out by the same
rules:

* ``eet_plurality`` — plain vote count, bucketed by
  :func:`canonical_answer`.
* ``eet_zsoftmax_tail`` — the weighted vote: within-pool z-scored tail
  confidence turned into ``softmax(beta * z)`` weights, summed per
  bucket.
* ``eet_max_tail_conf`` — the answer of the single highest tail
  confidence trace.

All three read the SAME per-trace scalar, ``readout_tail_conf``: the
mean per-token confidence over the last ``window_size`` tokens of the
trace, with the per-token confidence taken over the top
:data:`~.utils.READOUT_TOPK` logprobs. That is byte-for-byte the
definition of ``TraceInfo.tail_conf`` in ExploreExploitThink
(``conf_full_mean`` at ``branch_scoring_metric_k`` over
``group_conf_window`` tokens), which is why the sampling params request
``READOUT_TOPK`` logprobs while DeepConf's own confidence keeps slicing
its published top-20.

Every function here is a copy of the ExploreExploitThink original —
``src/eet/answers.py`` and ``src/eet/results.py``. Keep them in sync;
they are the comparison's fairness guarantee, not local helpers.
"""
import re
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Inverse temperature of the z-softmax weighted vote (eet.results.ZSOFTMAX_BETA).
ZSOFTMAX_BETA = 1.0


def quick_parse(text: str) -> str:
    """Strip ``\\text{...}`` wrappers — DeepConf's own prediction-side parse."""
    if '\\text{' in text and '}' in text:
        while '\\text{' in text:
            start = text.find('\\text{')
            if start == -1:
                break
            end = text.find('}', start)
            if end == -1:
                break
            content = text[start + 6:end]
            text = text[:start] + content + text[end + 1:]
    return text


def canonical_answer(answer: str) -> str:
    r"""Vote-bucketing key: two answers share a bucket exactly when the
    grader (``equal_func``) treats them identically.

    Copy of ``eet.answers.canonical_answer``. Every transform is a
    verified grader identity: ``quick_parse`` is the grader's own
    ``\text{}`` strip; ``\dfrac``/``\tfrac`` parse as ``\frac``;
    whitespace is dropped only where it touches a non-alphanumeric
    character; single letters uppercase to match case-insensitive mcqa
    grading. Comma lists are NOT reordered — ``math_equal`` is
    order-sensitive on them.
    """
    s = quick_parse(answer).strip()
    s = s.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r" (?![0-9A-Za-z])", "", s)
    s = re.sub(r"(?<![0-9A-Za-z]) ", "", s)
    if len(s) == 1 and s.isalpha():
        return s.upper()
    return s


def plurality_vote(
    answers: List[Optional[str]],
) -> Tuple[Optional[str], int, int]:
    """Canonicalized vote count.

    Returns ``(answer, leader_votes, n_voters)``; the answer is the
    winning bucket's first raw string (ties break by first appearance).
    """
    counts: Dict[str, int] = {}
    first_raw: Dict[str, str] = {}
    n = 0
    for a in answers:
        if a is None:
            continue
        n += 1
        key = canonical_answer(a)
        counts[key] = counts.get(key, 0) + 1
        first_raw.setdefault(key, a)
    if not counts:
        return None, 0, 0
    key = max(counts, key=counts.get)
    return first_raw[key], counts[key], n


def zsoftmax_vote(
    answers: List[Optional[str]], values: List[Optional[float]],
) -> Tuple[Optional[str], float, List[Optional[float]]]:
    """Canonicalized z-softmax weighted vote over one pool.

    Copy of ``eet.results.zsoftmax_vote``. Voters are the positions with
    a non-None answer AND a finite value; their values are z-scored
    within the pool (population std, ``sigma = max(sigma, 1e-6)``) and
    turned into ``softmax(beta * z)`` weights that sum to 1, summed per
    :func:`canonical_answer` bucket.

    Returns ``(answer, share, weights)`` — the winning bucket's first
    raw answer, its share of the total weight, and the per-position
    weights aligned with the inputs (``None`` for non-voters).
    """
    weights: List[Optional[float]] = [None] * len(answers)
    idx = [
        i for i, (a, v) in enumerate(zip(answers, values))
        if a is not None and v is not None and np.isfinite(v)
    ]
    if not idx:
        return None, 0.0, weights
    s = np.asarray([values[i] for i in idx], dtype=float)
    sigma_safe = max(float(s.std()), 1e-6)
    z = (s - s.mean()) / sigma_safe
    scaled = ZSOFTMAX_BETA * z
    scaled = scaled - scaled.max()
    exp_z = np.exp(scaled)
    w = exp_z / exp_z.sum()
    totals: Dict[str, float] = defaultdict(float)
    first_raw: Dict[str, str] = {}
    for i, wi in zip(idx, w):
        weights[i] = float(wi)
        key = canonical_answer(answers[i])
        totals[key] += float(wi)
        first_raw.setdefault(key, answers[i])
    winner = max(totals, key=totals.get)
    return first_raw[winner], totals[winner], weights


def argmax_vote(
    answers: List[Optional[str]], values: List[Optional[float]],
) -> Tuple[Optional[str], Optional[float], int]:
    """Answer of the single answered trace with the highest value.

    Copy of ``eet.results.argmax_answer("tail_conf")``: ties break to the
    earliest such trace in list order, traces with a None answer or a
    None value are skipped.

    Note the asymmetry with :func:`zsoftmax_vote`, which additionally
    drops non-finite values: the argmax does NOT, so an inf wins and a
    nan encountered first blocks every later trace (``v > nan`` is
    False). That is what the original does, and a read-out that "fixed"
    it here would stop being a mirror. It cannot bite in practice —
    ``readout_tail_conf`` is a mean of finite logprobs, 0.0 for an empty
    trace.
    """
    best_answer: Optional[str] = None
    best_value: Optional[float] = None
    n = 0
    for a, v in zip(answers, values):
        if a is None or v is None:
            continue
        n += 1
        if best_value is None or v > best_value:
            best_answer, best_value = a, float(v)
    return best_answer, best_value, n


def eet_voting_results(traces: List[Dict[str, Any]]) -> Dict[str, Any]:
    """The three ExploreExploitThink read-outs over one pool of traces.

    ``traces`` are the traces that DeepConf itself lets vote — for
    online mode ``DeepThinkOutput.all_voting_traces`` (warmup traces
    above ``conf_bar`` plus final traces not killed by the threshold),
    for offline mode every trace. The pool membership is DeepConf's
    decision; only the read-out is ours.

    Returns entries in the same shape as
    ``utils.compute_all_voting_results`` so they grade through
    ``evaluate_voting_results`` unchanged.
    """
    answers = [t.get('extracted_answer') or None for t in traces]
    values = [t.get('readout_tail_conf') for t in traces]

    plur_ans, plur_votes, n_voters = plurality_vote(answers)
    zs_ans, zs_share, _ = zsoftmax_vote(answers, values)
    am_ans, am_value, am_n = argmax_vote(answers, values)

    return {
        'eet_plurality': {
            'answer': plur_ans,
            'num_votes': n_voters,
            'confidence': plur_votes / n_voters if n_voters else None,
        },
        'eet_zsoftmax_tail': {
            'answer': zs_ans,
            'num_votes': n_voters,
            'confidence': zs_share,
        },
        'eet_max_tail_conf': {
            'answer': am_ans,
            'num_votes': am_n,
            'confidence': am_value,
        },
    }
