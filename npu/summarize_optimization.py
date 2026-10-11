"""Aggregate the preserved diagnosis and four-rank training optimization receipts."""
import hashlib
import json
from pathlib import Path
import statistics


def main():
    root = Path(__file__).resolve().parent
    path = root / 'evidence/optimization001.json'
    data = json.loads(path.read_text())
    files = data['files']
    def read(path): return json.loads(files[path])
    cache = '/cache/pi05-npu-20261011/optimize001'
    run = '/home/ma-user/pi05-npu-20261011/optimize001'
    result = read(run + '/result.json')
    assert result['exit_code'] == 0
    completions = [read(f'{cache}/rank-{r}-complete.json') for r in range(4)]
    assert all(c['status'] == 'benchmark_complete' and c['updates'] == 77 for c in completions)
    blocks = {}
    for label in completions[0]['variants']:
        reports = [read(f'{cache}/rank-{r}-{label}.json') for r in range(4)]
        assert all(len(r['rows']) == 8 for r in reports)
        seconds = []
        for rows in zip(*(r['rows'] for r in reports)):
            assert len({r['step'] for r in rows}) == 1
            seconds.append(max(r['seconds'] for r in rows))
        median = statistics.median(seconds)
        blocks[label] = {'seconds': seconds, 'median_seconds': median, 'mean_seconds': statistics.mean(seconds),
                         'min_seconds': min(seconds), 'max_seconds': max(seconds),
                         'median_samples_per_second': 4 / median,
                         'peak_GiB': max(r['peak_GiB'] for report in reports for r in report['rows']),
                         'bucket_MiB': reports[0]['bucket_MiB'], 'optimizer': reports[0]['optimizer'],
                         'optimizer_fused_flag': reports[0]['optimizer_fused_flag'],
                         'optimizer_step_devices': reports[0]['optimizer_step_devices'],
                         'ddp_rank0': reports[0]['ddp'],
                         'moments_finite_all_ranks': all(r['all_moments_finite'] for r in reports),
                         'gradient_sync_all_ranks': all(r['sampled_cross_rank_gradient_max_abs'] == 0 for r in reports)}
    numerics = {}
    for attempt in ['diagnose001', 'optimize001']:
        prefix = f'/cache/pi05-npu-20261011/{attempt}/'
        reports = {p[len(prefix):]: json.loads(value) for p, value in files.items()
                   if p.startswith(prefix) and p.endswith('.json')
                   and ('deterministic_' in p or 'bucket_numeric_' in p or 'optimizer_numerics' in p)}
        numerics[attempt] = reports
    summary = {'status': 'completed', 'result': result, 'completion': completions,
               'evidence_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
               'method': '3 warmup + 8 measured per variant; per-step slowest of four ranks, then median; no outliers removed',
               'HCCL_DETERMINISTIC': 'true', 'torch_deterministic_during_timing': False,
               'blocks': blocks, 'numerics': numerics, 'npu_smi_after': data['npu_smi'],
               'optimizer_probes': [read(f'{cache}/rank-{r}-optimizer_probes.json') for r in range(4)],
               'limits': ['eight cached LIBERO windows; full model, global batch four; no DataLoader/video decoding',
                          'sequential training blocks with continuing weights and optimizer states',
                          'bucket/checkpoint comparisons cover all active gradient elements on four inputs',
                          'optimizer numerical probe samples first 4096 elements per tensor and three updates; no convergence claim',
                          'original image augmentation remains outside torch deterministic mode due to NPU interpolate incompatibility']}
    (root / 'optimization_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps({k: {n: v[n] for n in ['median_seconds','mean_seconds','min_seconds','max_seconds','peak_GiB']}
                      for k, v in blocks.items()}, indent=2))


if __name__ == '__main__': main()
