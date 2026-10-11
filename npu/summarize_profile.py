"""Summarize preserved four-rank benchmark receipts; never discard slow samples."""
import hashlib
import json
from pathlib import Path
import statistics


def main():
    root = Path(__file__).resolve().parent
    evidence = root / 'evidence/profile002.json'
    data = json.loads(evidence.read_text())
    files = data['files']
    cache = '/cache/pi05-npu-20261011/profile002'
    run = '/home/ma-user/pi05-npu-20261011/profile002'
    def read(path):
        return json.loads(files[path])
    result = read(run + '/result.json')
    assert result['exit_code'] == 0
    complete = [read(f'{cache}/rank-{r}-complete.json') for r in range(4)]
    assert all(c['updates'] == 26 and c['status'] == 'profile_complete' for c in complete)
    timing = [read(f'{cache}/rank-{r}-timing_complete.json')['rows'] for r in range(4)]
    assert all(len(rows) == 25 for rows in timing)
    blocks = {}
    for label in ['baseline', 'layer_checkpoint_only', 'baseline_repeat', 'validation_overhead']:
        selected = [[row for row in rows if row['label'] == label] for rows in timing]
        assert len({len(rows) for rows in selected}) == 1
        seconds = []
        for aligned in zip(*selected):
            assert len({row['step'] for row in aligned}) == 1
            seconds.append(max(row['seconds'] for row in aligned))
        median = statistics.median(seconds)
        blocks[label] = {'slowest_rank_seconds_per_step': seconds, 'median_seconds': median,
                         'mean_seconds': statistics.mean(seconds), 'min_seconds': min(seconds),
                         'max_seconds': max(seconds), 'global_samples_per_second_at_median': 4 / median,
                         'peak_allocated_GiB': max(row['peak_GiB'] for rows in selected for row in rows)}
    breakdown = {}
    for label in ['baseline_breakdown', 'candidate_breakdown', 'audit_breakdown']:
        rows = [read(f'{cache}/rank-{rank}-{label}.json') for rank in range(4)]
        breakdown[label] = {'seconds_max_rank': max(r['seconds'] for r in rows),
                            'stage_seconds_median_across_ranks': {
                                key: statistics.median(r['stages_seconds'][key] for r in rows)
                                for key in rows[0]['stages_seconds']}}
    base = blocks['baseline']['median_seconds']
    candidate = blocks['layer_checkpoint_only']['median_seconds']
    summary = {'status': 'profile_complete', 'result': result,
               'evidence_sha256': hashlib.sha256(evidence.read_bytes()).hexdigest(),
               'aggregation': 'per-step maximum duration across four ranks, then median; no outliers removed',
               'blocks': blocks, 'breakdown': breakdown,
               'candidate_step_time_reduction_fraction': 1 - candidate / base,
               'candidate_throughput_increase_fraction': base / candidate - 1,
               'numerical_comparisons': [read(f'{cache}/rank-{r}-equivalence.json') for r in range(4)],
               'completion': complete, 'trace': read(cache + '/trace_summary.json'),
               'npu_smi_after': data['npu_smi'],
               'limitations': ['Short cached-data benchmark, not full DataLoader/video decoding or long-run convergence',
                              'Sequential timing blocks use continuing weights; fixed-state numerical comparisons are separate',
                              'Candidate failed strict gradient sample match; baseline repeat also differs; not promoted',
                              'Single-rank profiler capture has stop-state and two incomplete memory-record warnings',
                              'Profiler overhead excluded; trace durations are not steady-state wall-time fractions']}
    (root / 'profile_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps({'blocks': blocks, 'breakdown': breakdown,
                      'step_time_reduction': summary['candidate_step_time_reduction_fraction']}, indent=2))


if __name__ == '__main__':
    main()
