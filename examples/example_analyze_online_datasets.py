"""
Analyzer for DeepThinkLLM online mode results — dataset / per-problem variant.

Differences vs. example_analyze_online.py:
  1. Drops the noisy "Incorrect answer found: \\text{...}" spam.
  2. Lists every problem the primary voting method got wrong (including
     non-answers), showing the chosen consensus answer for each.
  3. ALL accuracy denominators include non-answers — traces (or methods)
     that did not produce an extracted answer count as incorrect, instead of
     being silently dropped from the denominator. This fixes the inflated
     metric that ships with the upstream DeepConf code.
  4. Accuracies are printed as both fraction (correct/total) and percentage.

python examples/example_analyze_online_datasets.py --output_dir ./online-dpsk/ --max_qid 29 --rids 1
"""
import os
import pickle
import argparse
from pathlib import Path
from collections import defaultdict
import numpy as np
from typing import Dict, List, Tuple, Any
import sys
from tqdm import tqdm


def find_result_files(output_dir: str, max_qid: int = None, rids: List[str] = None) -> List[Path]:
    """Find all result pickle files in the output directory."""
    output_path = Path(output_dir)
    pkl_files = list(output_path.glob("deepthink_online_*.pkl"))

    if max_qid is not None:
        pkl_files = [f for f in pkl_files if any(f"qid{qid}_" in f.name for qid in range(max_qid + 1))]
    if rids:
        pkl_files = [f for f in pkl_files if any(f"rid{rid}_" in f.name for rid in rids)]

    return sorted(pkl_files)


def extract_qid_rid(filename: str) -> Tuple[int, str]:
    """Extract (qid, rid) from filename `deepthink_online_qid{qid}_rid{rid}_<ts>.pkl`."""
    import re
    match = re.search(r"deepthink_online_qid(\d+)_rid([^_]+)_", filename)
    if match:
        return int(match.group(1)), match.group(2)
    return None, None


def load_result(filepath: Path) -> Dict:
    try:
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None


def check_missing_files(output_dir: str, max_qid: int, rids: List[str]) -> Dict:
    output_path = Path(output_dir)
    if not output_path.exists():
        return {
            'total_expected': (max_qid + 1) * len(rids),
            'total_found': 0,
            'missing_count': (max_qid + 1) * len(rids),
            'missing_pairs': [(qid, rid) for qid in range(max_qid + 1) for rid in rids],
            'existing_pairs': []
        }

    existing_files = find_result_files(output_dir)
    existing_pairs = set()
    file_map = {}
    for filepath in existing_files:
        qid, rid = extract_qid_rid(filepath.name)
        if qid is not None and rid is not None:
            existing_pairs.add((qid, rid))
            file_map[(qid, rid)] = filepath.name

    missing_pairs = []
    for qid in range(max_qid + 1):
        for rid in rids:
            if (qid, rid) not in existing_pairs:
                missing_pairs.append((qid, rid))

    return {
        'total_expected': (max_qid + 1) * len(rids),
        'total_found': len(existing_pairs),
        'missing_count': len(missing_pairs),
        'missing_pairs': sorted(missing_pairs),
        'existing_pairs': sorted(existing_pairs),
        'file_map': file_map,
    }


def analyze_token_usage(results: List[Dict]) -> Dict:
    token_stats = {
        'total_tokens': [],
        'warmup_tokens': [],
        'final_tokens': [],
        'tokens_per_trace': [],
    }
    for result in results:
        if not result:
            continue
        token_stats['total_tokens'].append(result['token_stats'].get('total_tokens', 0))
        token_stats['warmup_tokens'].append(result['token_stats'].get('warmup_tokens', 0))
        token_stats['final_tokens'].append(result['token_stats'].get('final_tokens', 0))
        total_traces = len(result.get('warmup_traces', [])) + len(result.get('final_traces', []))
        if total_traces > 0 and result['token_stats'].get('total_tokens', 0) > 0:
            token_stats['tokens_per_trace'].append(result['token_stats']['total_tokens'] / total_traces)

    stats = {}
    for key, values in token_stats.items():
        if values:
            stats[key] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'median': np.median(values),
                'total': np.sum(values),
            }
    return stats


def analyze_timing_details(results: List[Dict]) -> Dict:
    timing_components = {
        'total_time': [],
        'tokenizer_init_time': [],
        'llm_init_time': [],
        'warmup_gen_time': [],
        'warmup_process_time': [],
        'final_gen_time': [],
        'final_process_time': [],
        'generation_time': [],
        'processing_time': [],
    }
    derived_metrics = {
        'warmup_total_time': [],
        'final_total_time': [],
        'init_total_time': [],
        'inference_time': [],
        'throughput_tokens_per_sec': [],
        'warmup_throughput': [],
        'final_throughput': [],
    }

    for result in results:
        if not result:
            continue
        timing_stats = result.get('timing_stats', {})
        token_stats = result.get('token_stats', {})

        for component in timing_components:
            if component == 'generation_time':
                timing_components['generation_time'].append(
                    timing_stats.get('warmup_gen_time', 0) + timing_stats.get('final_gen_time', 0)
                )
                continue
            timing_components[component].append(timing_stats.get(component, 0))

        warmup_gen = timing_stats.get('warmup_gen_time', 0)
        warmup_proc = timing_stats.get('warmup_process_time', 0)
        final_gen = timing_stats.get('final_gen_time', 0)
        final_proc = timing_stats.get('final_process_time', 0)
        tokenizer_init = timing_stats.get('tokenizer_init_time', 0)
        llm_init = timing_stats.get('llm_init_time', 0)
        total_time = timing_stats.get('total_time', 0)

        derived_metrics['warmup_total_time'].append(warmup_gen + warmup_proc)
        derived_metrics['final_total_time'].append(final_gen + final_proc)
        derived_metrics['init_total_time'].append(tokenizer_init + llm_init)
        derived_metrics['inference_time'].append(max(0, total_time - tokenizer_init - llm_init))

        total_tokens = token_stats.get('total_tokens', 0)
        warmup_tokens = token_stats.get('warmup_tokens', 0)
        final_tokens = token_stats.get('final_tokens', 0)

        total_gen_time = warmup_gen + final_gen
        derived_metrics['throughput_tokens_per_sec'].append(
            total_tokens / total_gen_time if total_gen_time > 0 and total_tokens > 0 else 0
        )
        derived_metrics['warmup_throughput'].append(
            warmup_tokens / warmup_gen if warmup_gen > 0 and warmup_tokens > 0 else 0
        )
        derived_metrics['final_throughput'].append(
            final_tokens / final_gen if final_gen > 0 and final_tokens > 0 else 0
        )

    all_timing_data = {**timing_components, **derived_metrics}
    timing_stats = {}
    for key, values in all_timing_data.items():
        if not values:
            continue
        if 'throughput' in key.lower():
            values = [v for v in values if v > 0]
        if not values:
            continue
        timing_stats[key] = {
            'mean': np.mean(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values),
            'median': np.median(values),
            'total': np.sum(values),
            'count': len(values),
        }
    return timing_stats


def analyze_voting_methods(results: List[Dict]) -> Dict:
    """Voting-method accuracy where the denominator includes non-answers.

    Every loaded result counts toward `total` for every method that appears in
    any result's evaluation. If a method did not produce an answer for a given
    problem (evaluation missing, method missing, answer is None, or
    is_correct is False) it counts as incorrect.
    """
    # First pass: union of method names across all results.
    all_methods = []
    seen = set()
    for r in results:
        if r and r.get('evaluation'):
            for method in r['evaluation'].keys():
                if method not in seen:
                    seen.add(method)
                    all_methods.append(method)

    # Second pass: per-method correct/total + auxiliary stats over only
    # the results that actually produced an answer for that method.
    method_accuracy = {}
    n_results = sum(1 for r in results if r is not None)
    for method in all_methods:
        correct = 0
        confidences = []
        num_votes_list = []
        for r in results:
            if not r:
                continue
            eval_data = (r.get('evaluation') or {}).get(method)
            if eval_data and eval_data.get('is_correct'):
                correct += 1
            if eval_data and eval_data.get('answer') is not None:
                if eval_data.get('confidence') is not None:
                    confidences.append(eval_data['confidence'])
                if eval_data.get('num_votes') is not None:
                    num_votes_list.append(eval_data['num_votes'])
        total = n_results  # denominator includes non-answers
        method_accuracy[method] = {
            'accuracy': correct / total if total > 0 else 0.0,
            'correct': correct,
            'total': total,
            'avg_confidence': float(np.mean(confidences)) if confidences else None,
            'num_votes': float(np.mean(num_votes_list)) if num_votes_list else 0.0,
        }
    return method_accuracy


def _true_confidence_denominator(result: Dict, method: str) -> int:
    """Compute the true (non-answer-inclusive) denominator for one result and one
    confidence-based bucket, by re-applying the bucket filter to raw traces.

    The upstream code adds an `extracted_answer` filter to each of these
    bucket definitions; we drop it here so traces with no extracted answer
    still count toward the denominator (as incorrect).
    """
    warmup_traces = result.get('warmup_traces') or []
    final_traces = result.get('final_traces') or []
    conf_bar = result.get('conf_bar')

    if method == 'warmup_above_threshold':
        if conf_bar is None:
            return 0
        return sum(1 for t in warmup_traces if t.get('min_conf', 0) >= conf_bar)
    if method == 'final_completed':
        return sum(1 for t in final_traces if t.get('stop_reason') != 'gconf_threshold')
    if method == 'final_completed_notmaxtokenshit':
        return sum(1 for t in final_traces if t.get('stop_reason') not in ('gconf_threshold', 'length'))
    if method == 'early_stopped':
        return sum(1 for t in final_traces if t.get('stop_reason') == 'gconf_threshold')
    # Unknown bucket: fall back to the pre-computed total (will be inflated).
    return -1


def analyze_confidence_methods(results: List[Dict]) -> Dict:
    """Per-trace confidence-bucket accuracy with non-answers counted as wrong.

    Strategy: reuse the precomputed `confidence_evaluation[method].correct`
    count from each result (still valid — non-answers can only contribute 0
    to correct), but recompute the denominator from raw traces so it
    includes traces that never produced an extracted answer.
    """
    conf_stats = defaultdict(lambda: {'correct': 0, 'total': 0})

    for result in results:
        if not result:
            continue
        ce = result.get('confidence_evaluation') or {}
        # We iterate over the set of bucket names that EITHER appeared in this
        # result's confidence_evaluation OR are known buckets, so we don't
        # silently drop a bucket whose recomputed denominator is positive but
        # whose precomputed correct count is 0 (i.e., the upstream filter
        # made it absent from confidence_evaluation entirely).
        known_buckets = (
            'warmup_above_threshold',
            'final_completed',
            'final_completed_notmaxtokenshit',
            'early_stopped',
        )
        for method in set(known_buckets) | set(ce.keys()):
            correct = ce.get(method, {}).get('correct', 0)
            true_total = _true_confidence_denominator(result, method)
            if true_total < 0:
                # Unknown bucket — fall back to whatever upstream recorded.
                true_total = ce.get(method, {}).get('total', 0)
            conf_stats[method]['correct'] += correct
            conf_stats[method]['total'] += true_total

    conf_accuracy = {}
    for method, stats in conf_stats.items():
        if stats['total'] > 0:
            conf_accuracy[method] = {
                'accuracy': stats['correct'] / stats['total'],
                'correct': stats['correct'],
                'total': stats['total'],
            }
    return conf_accuracy


def analyze_overall_trace_accuracy(results: List[Dict]) -> Dict:
    """Aggregate per-trace accuracy across all results, counting every trace
    in `all_voting_traces` (with no `extracted_answer` filter) in the denominator.

    `all_voting_traces` is not always preserved in the pkl; if missing we
    approximate by summing `warmup_traces` + `final_traces` and counting any
    trace with an extracted answer that matches the ground truth as correct.
    """
    total = 0
    correct = 0
    for r in results:
        if not r:
            continue
        gt = r.get('ground_truth')
        if gt is None:
            continue
        # Strip a \boxed{...} wrapper if present (matches example_online.py).
        gt_stripped = _strip_boxed(gt)
        traces = r.get('all_voting_traces')
        if traces is None:
            traces = list(r.get('warmup_traces') or []) + list(r.get('final_traces') or [])
        for t in traces:
            total += 1
            ans = t.get('extracted_answer')
            if ans is None:
                continue
            if _simple_equal(ans, gt_stripped):
                correct += 1
    if total == 0:
        return {}
    return {'correct': correct, 'total': total, 'accuracy': correct / total}


def _strip_boxed(text: str) -> str:
    """Best-effort \\boxed{...} extraction without pulling in deepconfTesting.utils."""
    import re
    s = str(text)
    if 'boxed' not in s:
        return s.strip()
    # Find the first 'boxed' and walk braces.
    i = s.find('boxed') + len('boxed')
    if i >= len(s) or s[i] != '{':
        return s[i:].split('$')[0].strip()
    stack, out = 1, []
    for c in s[i + 1:]:
        if c == '{':
            stack += 1
            out.append(c)
        elif c == '}':
            stack -= 1
            if stack == 0:
                break
            out.append(c)
        else:
            out.append(c)
    return ''.join(out).strip()


def _simple_equal(answer: str, ground_truth: str) -> bool:
    """Mirror of example_online.py:equal_func for the single-letter MCQA path.

    Strips a leading `\\text{...}` wrapper and does case-insensitive
    single-letter compare; falls back to string equality otherwise. Math
    benchmarks should prefer the precomputed correct counts in
    `confidence_evaluation`; this helper is only used for the aggregate
    per-trace accuracy.
    """
    import re
    a = str(answer)
    # Strip \text{...}
    while '\\text{' in a:
        start = a.find('\\text{')
        end = a.find('}', start)
        if end == -1:
            break
        a = a[:start] + a[start + 6:end] + a[end + 1:]
    a = a.strip()
    g = str(ground_truth).strip()
    if len(a) == 1 and a.isalpha() and len(g) == 1 and g.isalpha():
        return a.lower() == g.lower()
    return a == g


def print_incorrect_problems(results: List[Dict], primary_method: str = 'majority'):
    """List every problem the `primary_method` got wrong, including non-answers.

    For each one, print the consensus answer that the method chose (or
    `<no answer>` if the method produced no answer or the result had no
    evaluation block at all).
    """
    if not results:
        return

    rows = []
    for r in results:
        if not r:
            continue
        eval_data = (r.get('evaluation') or {}).get(primary_method)
        if eval_data is None:
            answer, is_correct, tag = None, False, 'no-answer'
        else:
            answer = eval_data.get('answer')
            is_correct = bool(eval_data.get('is_correct'))
            tag = 'correct' if is_correct else ('no-answer' if answer is None else 'incorrect')
        if is_correct:
            continue  # only listing incorrect / non-answer problems
        rows.append((r.get('qid'), r.get('run_id'), r.get('ground_truth'), answer, tag))

    print(f"\n🚫 Incorrect Problems (method: {primary_method}): {len(rows)}")
    print("-" * 70)
    if not rows:
        print("  (none — every problem was answered correctly)")
        return

    rows.sort(key=lambda x: (str(x[1]), x[0] if x[0] is not None else -1))
    for qid, rid, gt, answer, tag in rows:
        ans_display = '<no answer>' if answer is None else str(answer)
        if len(ans_display) > 40:
            ans_display = ans_display[:39] + '…'
        print(f"  qid={qid} rid={rid}: gt={gt}, consensus={ans_display}  [{tag}]")


def print_timing_breakdown(timing_stats: Dict):
    print(f"\n⏱️ Detailed Timing Analysis")
    print("=" * 80)
    categories = {
        'Generation Times': ['warmup_gen_time', 'final_gen_time', 'generation_time'],
    }
    for category, metrics in categories.items():
        print(f"\n📊 {category}")
        print("-" * 60)
        print(f"{'Metric':<25} {'Mean ± Std':<18} {'Median':<10} {'Range':<15}")
        print("-" * 70)
        for metric in metrics:
            if metric not in timing_stats:
                continue
            stats = timing_stats[metric]
            mean_str = f"{stats['mean']:.2f}s ± {stats['std']:.2f}s"
            median_str = f"{stats['median']:.2f}s"
            range_str = f"[{stats['min']:.2f}s, {stats['max']:.2f}s]"
            display_name = metric.replace('_', ' ').title()
            if len(display_name) > 24:
                display_name = display_name[:21] + "..."
            print(f"{display_name:<25} {mean_str:<18} {median_str:<10} {range_str:<15}")


def _fmt_acc(correct: int, total: int) -> str:
    if total <= 0:
        return f"{correct}/{total} (n/a)"
    pct = 100.0 * correct / total
    return f"{correct}/{total} ({pct:.1f}%)"


def print_statistics(token_stats: Dict, method_accuracy: Dict, conf_accuracy: Dict,
                     trace_accuracy: Dict, missing_info: Dict, results: List[Dict],
                     timing_stats: Dict = None):
    print("\n" + "=" * 80)
    print("DEEPTHINK RESULTS ANALYSIS (denominator includes non-answers)")
    print("=" * 80)

    print(f"\n📊 Overall Statistics")
    print("-" * 40)
    print(f"Total result files analyzed: {len(results)}")
    valid_results = sum(1 for r in results if r is not None)
    print(f"Valid results: {valid_results}")

    if token_stats:
        print(f"\n💰 Token Usage Statistics")
        print("-" * 40)
        for token_type, stats in token_stats.items():
            print(f"\n{token_type.replace('_', ' ').title()}:")
            print(f"  Mean: {stats['mean']:,.0f} ± {stats['std']:,.0f}")
            print(f"  Median: {stats['median']:,.0f}")
            print(f"  Range: [{stats['min']:,.0f}, {stats['max']:,.0f}]")
            if 'total' in stats:
                print(f"  Total: {stats['total']:,.0f}")

    if timing_stats:
        print_timing_breakdown(timing_stats)

    if method_accuracy:
        print(f"\n🗳️ Voting Methods Accuracy")
        print("-" * 40)
        header = f"{'Method':<30} {'Correct/Total (Pct)':<22} {'Avg Conf':<10} {'Num Votes':<10}"
        print(header)
        print("-" * len(header))
        for method, stats in sorted(method_accuracy.items(), key=lambda x: x[1]['accuracy'], reverse=True):
            acc_str = _fmt_acc(stats['correct'], stats['total'])
            conf_str = f"{stats['avg_confidence']:.3f}" if stats['avg_confidence'] is not None else "N/A"
            num_votes = f"{stats['num_votes']:.3f}"
            print(f"{method:<30} {acc_str:<22} {conf_str:<10} {num_votes:<10}")

    if conf_accuracy:
        print(f"\n🎯 Confidence-Based Methods")
        print("-" * 40)
        header = f"{'Method':<35} {'Correct/Total (Pct)':<22}"
        print(header)
        print("-" * len(header))
        for method, stats in sorted(conf_accuracy.items(), key=lambda x: x[1]['accuracy'], reverse=True):
            print(f"{method:<35} {_fmt_acc(stats['correct'], stats['total']):<22}")

    if trace_accuracy:
        print(f"\n🧵 Overall Per-Trace Accuracy (all_voting_traces or warmup+final)")
        print("-" * 40)
        print(f"  {_fmt_acc(trace_accuracy['correct'], trace_accuracy['total'])}")

    print(f"\n📁 File Coverage Analysis")
    print("-" * 40)
    print(f"Expected files: {missing_info['total_expected']}")
    print(f"Found files: {missing_info['total_found']}")
    print(f"Missing files: {missing_info['missing_count']}")
    coverage = (missing_info['total_found'] / missing_info['total_expected'] * 100
                if missing_info['total_expected'] > 0 else 0)
    print(f"Coverage: {coverage:.1f}%")

    if missing_info['missing_pairs']:
        print(f"\n⚠️ Missing (qid, rid) pairs:")
        by_rid = defaultdict(list)
        for qid, rid in missing_info['missing_pairs']:
            by_rid[rid].append(qid)
        for rid, qids in sorted(by_rid.items()):
            if len(qids) <= 10:
                print(f"  rid={rid}: qids={qids}")
            else:
                print(f"  rid={rid}: {len(qids)} missing qids (showing first 10): {qids[:10]}...")


def main():
    parser = argparse.ArgumentParser(description='Analyze DeepThinkLLM results (per-problem, non-answer-inclusive denominators)')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory containing output pickle files')
    parser.add_argument('--max_qid', type=int, required=True,
                        help='Maximum question ID (0-based)')
    parser.add_argument('--rids', type=str, nargs='+', required=True,
                        help='List of run IDs to check')
    parser.add_argument('--force', action='store_true',
                        help='Force analysis even if files are missing')
    parser.add_argument('--check_only', action='store_true',
                        help='Only check for missing files, do not analyze')
    parser.add_argument('--detailed_timing', action='store_true', default=True,
                        help='Enable detailed timing analysis (default: True)')
    parser.add_argument('--primary_method', type=str, default='majority',
                        help='Voting method whose consensus answer is reported for incorrect problems (default: majority)')
    parser.add_argument('--no_per_problem', action='store_true',
                        help='Suppress the list-of-incorrect-problems section')
    args = parser.parse_args()

    print("\n" + "=" * 80)
    print("PHASE 1: FILE COMPLETENESS CHECK")
    print("=" * 80)
    print(f"Directory: {args.output_dir}")
    print(f"Expected QIDs: 0 to {args.max_qid} (total: {args.max_qid + 1})")
    print(f"Expected RIDs: {args.rids} (total: {len(args.rids)})")
    print(f"Total expected files: {(args.max_qid + 1) * len(args.rids)}")

    missing_info = check_missing_files(args.output_dir, args.max_qid, args.rids)
    print("\n📊 Results:")
    print(f"  ✓ Found: {missing_info['total_found']} files")
    print(f"  ✗ Missing: {missing_info['missing_count']} files")
    coverage = (missing_info['total_found'] / missing_info['total_expected'] * 100
                if missing_info['total_expected'] > 0 else 0)
    print(f"  📈 Coverage: {coverage:.1f}%")

    if missing_info['missing_count'] > 0:
        print("\n" + "=" * 80)
        print("⚠️  MISSING FILES DETECTED")
        print("=" * 80)
        by_rid = defaultdict(list)
        for qid, rid in missing_info['missing_pairs']:
            by_rid[rid].append(qid)
        for rid, qids in sorted(by_rid.items()):
            print(f"\nRID '{rid}':")
            print(f"  Missing {len(qids)} files")
            if len(qids) <= 15:
                print(f"  QIDs: {sorted(qids)}")
            else:
                print(f"  QIDs (first 15): {sorted(qids)[:15]}...")
                print(f"  ... and {len(qids)-15} more")

        if args.check_only:
            print("\n✅ Check complete (--check_only flag used)")
            sys.exit(0)
        if not args.force:
            print("\n❌ ABORTING: Files are missing. Use --force to analyze incomplete data.")
            sys.exit(1)
        print("\n⚠️  WARNING: Continuing with incomplete data (--force used)")
    else:
        print("\n✅ All expected files are present!")
        if args.check_only:
            sys.exit(0)

    print("\n" + "=" * 80)
    print("PHASE 2: LOADING AND ANALYZING FILES")
    print("=" * 80)
    result_files = find_result_files(args.output_dir, max_qid=args.max_qid, rids=args.rids)
    print(f"Loading {len(result_files)} files...")

    results = []
    load_errors = []
    for i, filepath in tqdm(enumerate(result_files)):
        result = load_result(filepath)
        if result:
            results.append(result)
        else:
            load_errors.append(filepath.name)
    print(f"\n📊 Loading Summary:")
    print(f"  ✓ Successfully loaded: {len(results)} files")
    if load_errors:
        print(f"  ✗ Failed to load: {len(load_errors)} files")
        for err_file in load_errors[:5]:
            print(f"    - {err_file}")
        if len(load_errors) > 5:
            print(f"    ... and {len(load_errors)-5} more")

    if not results:
        print("\n❌ No valid results to analyze!")
        sys.exit(1)

    print("\n" + "=" * 80)
    print("PHASE 3: ANALYZING RESULTS")
    print("=" * 80)
    print("Running analyses (no per-problem print spam)...")
    token_stats = analyze_token_usage(results)
    method_accuracy = analyze_voting_methods(results)
    conf_accuracy = analyze_confidence_methods(results)
    trace_accuracy = analyze_overall_trace_accuracy(results)
    timing_stats = analyze_timing_details(results) if args.detailed_timing else None

    if not args.no_per_problem:
        print_incorrect_problems(results, primary_method=args.primary_method)

    print_statistics(token_stats, method_accuracy, conf_accuracy, trace_accuracy,
                     missing_info, results, timing_stats)

    print("\n✅ Analysis complete!")


if __name__ == "__main__":
    main()
