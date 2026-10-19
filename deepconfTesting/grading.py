"""Answer grading with a time limit: DeepConf's ``equal_func`` (dynasor's
``math_equal``) behind a persistent worker process.

This file is identical in ExploreExploitThink (``src/eet/grading.py``) and
deepconfTesting (``deepconfTesting/grading.py``) and imports nothing from
either package: run as a script, it is the worker. ``check_correct`` sends
each new (answer, ground truth) pair to the worker and waits at most
``GRADING_TIMEOUT_S``. A comparison that takes longer (sympy can spend
minutes simplifying an unevaluated sum) counts as not equal, as in
dynasor's own timeout mode; it is printed and listed in ``GRADER.timed_out``,
its worker is killed and the next pair starts a fresh one. Results are
memoized per process, which is exact because ``equal_func`` grades every
pair from latex2sympy2's import-time state, so a pair grades the same in
any order. The worker is started with ``subprocess`` (fork + exec), which is
safe inside a multithreaded process with CUDA initialized. The client is not
thread-safe; the runners grade on their main thread.
"""
import atexit
import json
import os
import select
import subprocess
import sys

import latex2sympy2
from dynasor.core.evaluator import math_equal

# The slowest correct pair of the model-screening pools took 0.06 s and the
# slowest comparison that finished 6.7 s, so the limit only cuts off hangs.
GRADING_TIMEOUT_S = 30.0

# latex2sympy2 (behind math_equal) keeps parse state at module level:
# convert_atom declares ``global var`` and rebinds it to a Symbol whenever it
# parses a differential ("d" + letter, e.g. an answer reading "0 and the three
# cube roots of 18"), after which every parse of a letter raises and
# math_equal falls back to string comparison for the rest of the process.
_LATEX2SYMPY_STATE = (dict(latex2sympy2.var), dict(latex2sympy2.variances))


def quick_parse(text: str) -> str:
    """Parse LaTeX text content (DeepConf's ``quick_parse``)."""
    if '\\text{' in text and '}' in text:
        # Find all occurrences of \text{...} and remove them
        while '\\text{' in text:
            start = text.find('\\text{')
            if start == -1:
                break
            end = text.find('}', start)
            if end == -1:
                break
            # Replace \text{content} with just content
            content = text[start + 6:end]  # 6 is length of '\text{'
            text = text[:start] + content + text[end + 1:]
    return text


def equal_func(answer: str, ground_truth: str) -> bool:
    """Check if answer equals ground truth (DeepConf's ``equal_func``).

    ``quick_parse`` is applied only to the answer (not the ground truth),
    and a case-insensitive single-letter alpha shortcut handles
    multiple-choice answers; everything else falls through to dynasor's
    ``math_equal``, started from latex2sympy2's import-time state. Runs in
    the worker; call ``check_correct`` instead.
    """
    latex2sympy2.var = dict(_LATEX2SYMPY_STATE[0])
    latex2sympy2.variances = dict(_LATEX2SYMPY_STATE[1])
    answer = quick_parse(answer)
    if len(answer) == 1 and answer.isalpha() and len(ground_truth) == 1 and ground_truth.isalpha():
        return answer.lower() == ground_truth.lower()
    return math_equal(answer, ground_truth)


class Grader:
    """Client of the grading worker: memo, deadline and timeout record."""

    def __init__(self):
        self.memo = {}
        self.timed_out = []
        self._worker = None

    def equal(self, answer: str, ground_truth: str) -> bool:
        key = (answer, ground_truth)
        if key not in self.memo:
            self.memo[key] = self._ask(answer, ground_truth)
        return self.memo[key]

    def close(self) -> None:
        if self._worker is not None:
            self._worker.kill()
            self._worker.wait()
            self._worker.stdin.close()
            self._replies.close()
            self._worker = None

    def _start(self) -> None:
        reply_fd, worker_fd = os.pipe()
        # PYTHONSAFEPATH keeps the worker's own directory (a package with
        # generic module names) off sys.path; Python < 3.11 (deepConfEnv's
        # 3.10) ignores it, and no module the worker imports collides.
        self._worker = subprocess.Popen(
            [sys.executable, os.path.abspath(__file__), str(worker_fd)],
            stdin=subprocess.PIPE, text=True, pass_fds=(worker_fd,),
            env={**os.environ, "PYTHONSAFEPATH": "1"})
        os.close(worker_fd)
        self._replies = os.fdopen(reply_fd)
        # Not under the deadline: imports from a cold filesystem can be slow.
        if self._replies.readline() != "ready\n":
            raise RuntimeError("grading worker did not start")

    def _ask(self, answer: str, ground_truth: str) -> bool:
        if self._worker is None:
            self._start()
        self._worker.stdin.write(json.dumps([answer, ground_truth]) + "\n")
        self._worker.stdin.flush()
        # One reply line per request, so the reader never buffers past it.
        if not select.select([self._replies], [], [], GRADING_TIMEOUT_S)[0]:
            self.close()
            self.timed_out.append([answer, ground_truth])
            print(f"GRADING TIMEOUT ({GRADING_TIMEOUT_S:.0f} s), counted as not equal: "
                  f"{answer!r} vs {ground_truth!r}", flush=True)
            return False
        reply = self._replies.readline()
        if reply not in ("1\n", "0\n"):
            raise RuntimeError(f"grading worker died on {answer!r} vs {ground_truth!r}")
        return reply == "1\n"


GRADER = Grader()
atexit.register(GRADER.close)


def check_correct(prediction: str | None, reference: str) -> bool:
    """``False`` on missing/empty predictions, otherwise ``equal_func`` in
    the worker (memoized, at most ``GRADING_TIMEOUT_S``).

    DeepConf filters empty/``None`` extracted answers at the call site
    (``if trace.get('extracted_answer')`` in example_online.py); this
    filter centralizes it so callers can pass results through uniformly.
    """
    if not prediction:
        return False
    return GRADER.equal(prediction, reference)


def serve(reply_fd: int) -> None:
    """Worker loop: one JSON ``[answer, ground_truth]`` line per request on
    stdin, one ``1``/``0`` reply line on ``reply_fd``. Replies avoid stdout,
    which stays the job log (the ANTLR runtime can print a version warning
    on every parse)."""
    replies = os.fdopen(reply_fd, "w")
    replies.write("ready\n")
    replies.flush()
    for line in sys.stdin:
        answer, ground_truth = json.loads(line)
        replies.write("1\n" if equal_func(answer, ground_truth) else "0\n")
        replies.flush()


if __name__ == "__main__":
    serve(int(sys.argv[1]))
