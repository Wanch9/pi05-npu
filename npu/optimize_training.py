"""Bounded four-NPU numerical diagnosis and training-only optimization experiments."""
import argparse
from contextlib import nullcontext
from datetime import timedelta
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import random
import statistics
import time
from types import SimpleNamespace

import numpy as np
import safetensors.torch
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch_npu

from openpi.models_pytorch.device import local_device
from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
from npu.validate_four_cards import observation, rng_state, restore_rng


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--mode', choices=['diagnose', 'benchmark', 'all'], required=True)
    parser.add_argument('--weights', type=Path, required=True)
    parser.add_argument('--data', type=Path, required=True)
    args = parser.parse_args()
    rank = int(os.environ['RANK'])
    assert int(os.environ['WORLD_SIZE']) == 4
    device = local_device()
    acc = torch.npu
    dist.init_process_group('hccl', timeout=timedelta(seconds=600))
    if rank == 0:
        args.output.mkdir(parents=True, exist_ok=False)
    dist.barrier()

    def emit(event, data):
        (args.output / f'rank-{rank}-{event}.json').write_text(json.dumps(data, indent=2))
        print(json.dumps({'rank': rank, 'event': event, **data}), flush=True)

    emit('identity', {'pid': os.getpid(), 'device': str(device), 'cards': os.environ.get('ASCEND_RT_VISIBLE_DEVICES'),
                     'mode': args.mode, 'world_size': 4, 'global_batch': 4,
                     'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                     'torch': torch.__version__, 'torch_npu': torch_npu.__version__,
                     'HCCL_DETERMINISTIC': os.environ.get('HCCL_DETERMINISTIC'),
                     'no_checkpoint_or_inference': True})
    torch.set_num_threads(4)
    torch.manual_seed(42); np.random.seed(42); random.seed(42); acc.manual_seed_all(42)

    # Probe actual installed optimizer implementations before any model mutation.
    if args.mode in {'diagnose', 'all'}:
        from torch_npu.optim import NpuFusedAdamW
        probes = {}
        for name, factory in [('npu_fused', NpuFusedAdamW),
                              ('torch_fused', lambda params, **kw: torch.optim.AdamW(params, fused=True, **kw))]:
            try:
                p = torch.nn.Parameter(torch.ones(64, device=device, dtype=torch.bfloat16))
                q = torch.nn.Parameter(torch.ones(32, device=device, dtype=torch.float32))
                p.grad = torch.full_like(p, .25); q.grad = torch.full_like(q, .125)
                optimizer = factory([p, q], lr=5e-5, betas=(.9, .95), eps=1e-8, weight_decay=1e-10)
                optimizer.step(); acc.synchronize()
                probes[name] = {'supported': True, 'finite': bool(torch.isfinite(p).all() and torch.isfinite(q).all()),
                                'fp32_parameter_changed': bool((q != 1).any())}
            except (TypeError, RuntimeError, ValueError) as exc:
                probes[name] = {'supported': False, 'error': str(exc)}
            finally:
                del p, q
                if 'optimizer' in locals(): del optimizer
        emit('optimizer_probes', probes)
        gc.collect(); acc.empty_cache()

    started = time.perf_counter()
    cfg = SimpleNamespace(pi05=True, paligemma_variant='gemma_2b', action_expert_variant='gemma_300m',
                          dtype='bfloat16', action_dim=32, action_horizon=10, max_token_len=200,
                          discrete_state_input=False, pytorch_compile_mode=None)
    module = PI0Pytorch(cfg)
    safetensors.torch.load_model(module, str(args.weights),
                                strict=True, device='cpu')
    module.to(device); module.gradient_checkpointing_enable(); module.train()
    model = DDP(module, device_ids=[device.index], find_unused_parameters=True, gradient_as_bucket_view=True)
    samples = torch.load(args.data / 'samples.pt', weights_only=False)
    named = dict(module.named_parameters())
    torch.manual_seed(42+rank); np.random.seed(42+rank); random.seed(42+rank); acc.manual_seed_all(42+rank)
    original_checkpoint = module._apply_checkpoint
    original_preprocess = module._preprocess_observation

    def compatible_preprocess(*values, **kwargs):
        # Deterministic interpolate decomposition requests an unsupported NPU memory format.
        # Keep the original non-trainable augmentation path and RNG, while diagnosing the model backward.
        deterministic = torch.are_deterministic_algorithms_enabled()
        if deterministic: torch.use_deterministic_algorithms(False)
        try:
            return original_preprocess(*values, **kwargs)
        finally:
            if deterministic: torch.use_deterministic_algorithms(True)

    module._preprocess_observation = compatible_preprocess

    def layer_only(func, *values, **kwargs):
        if func.__name__ == 'forward_func': return func(*values, **kwargs)
        return original_checkpoint(func, *values, **kwargs)

    def set_candidate(candidate):
        module._apply_checkpoint = layer_only if candidate else original_checkpoint

    acc.synchronize(); dist.barrier()
    emit('loaded', {'seconds': time.perf_counter()-started,
                    'parameters': sum(p.numel() for p in module.parameters()),
                    'trainable': sum(p.numel() for p in module.parameters() if p.requires_grad)})

    def full_compare(reference):
        """Every active gradient element; chunk sums on NPU FP32, final aggregation CPU FP64."""
        stats = []; names = []; elements = 0
        for name, p in named.items():
            if p.grad is None:
                assert name not in reference
                continue
            ref = reference[name].flatten(); current = p.grad.detach().flatten()
            elements += current.numel(); names.append(name)
            chunks = []
            for offset in range(0, current.numel(), 16*1024*1024):
                a = ref[offset:offset+16*1024*1024].float()
                b = current[offset:offset+16*1024*1024].float()
                delta = (a-b).abs()
                chunks.append(torch.stack([delta.max(), delta.square().sum(), a.square().sum(),
                                           (delta != 0).float().sum(),
                                           (delta > (1e-7 + 1e-5*a.abs())).float().sum()]))
            values = torch.stack(chunks)
            stats.append(torch.cat([values[:, 0].max().view(1), values[:, 1:].sum(dim=0)]))
        values = torch.stack(stats).cpu().double()
        assert torch.isfinite(values).all(), 'Non-finite gradient comparison'
        total = values[:, 1:].sum(dim=0)
        worst = sorted(zip(names, values.tolist()), key=lambda x: x[1][0], reverse=True)[:12]
        return {'elements': elements, 'parameter_tensors': len(names), 'max_abs': values[:, 0].max().item(),
                'relative_l2': math.sqrt(total[0].item()/max(total[1].item(), 1e-300)),
                'different_elements': int(total[2].item()), 'outside_tolerance': int(total[3].item()),
                'strict_match': bool(total[3].item() == 0), 'bitwise_match': bool(total[2].item() == 0),
                'largest_differences': [{'name': n, 'max_abs': v[0]} for n, v in worst]}

    if args.mode in {'diagnose', 'all'}:
        saved_rng = rng_state(acc)

        def backward(candidate, communicate):
            set_candidate(candidate); restore_rng(saved_rng, acc); model.zero_grad(set_to_none=True)
            with nullcontext() if communicate else model.no_sync():
                loss = model(observation(samples[rank], device), samples[rank]['actions'].unsqueeze(0).to(device)).mean()
                loss.backward()
            acc.synchronize()
            assert torch.isfinite(loss)
            return loss.item()

        # Stabilize DDP before comparisons; no optimizer updates anywhere in this mode.
        backward(False, True)
        for deterministic, communicate in [(False, True), (False, False), (True, False), (True, True)]:
            label = f'deterministic_{int(deterministic)}_communication_{int(communicate)}'
            torch.use_deterministic_algorithms(deterministic)
            dist.barrier()
            reference_loss = backward(False, communicate)
            reference = {n: p.grad.detach().clone() for n, p in named.items() if p.grad is not None}
            emit(label+'_reference', {'loss': reference_loss, 'gradient_tensors': len(reference),
                                      'deterministic': torch.are_deterministic_algorithms_enabled()})
            for comparison, candidate in [('baseline_repeat', False), ('layer_only', True), ('layer_only_repeat', True)]:
                value = backward(candidate, communicate)
                metrics = full_compare(reference)
                metrics.update(loss_reference=reference_loss, loss=value, loss_abs_diff=abs(value-reference_loss))
                emit(label+'_'+comparison, metrics)
            del reference
            model.zero_grad(set_to_none=True); gc.collect(); acc.empty_cache()
        torch.use_deterministic_algorithms(False)
        emit('diagnosis_complete', {'status': 'diagnosis_complete', 'optimizer_updates': 0,
                          'full_gradient_comparisons': 12, 'candidate_not_automatically_promoted': True})
    if args.mode in {'benchmark', 'all'}:
        def rewrap(bucket):
            nonlocal model
            dist.barrier(); acc.synchronize(); model.zero_grad(set_to_none=True)
            model._remove_autograd_hooks()
            del model
            gc.collect(); acc.empty_cache()
            model = DDP(module, device_ids=[device.index], find_unused_parameters=True,
                        gradient_as_bucket_view=True, bucket_cap_mb=bucket)

        # Fixed-state bucket comparisons are separate from timed optimizer updates.
        torch.use_deterministic_algorithms(True)
        set_candidate(True)
        saved_rng = rng_state(acc)
        reference = None
        for bucket in [25, 50, 100, 25]:
            rewrap(bucket)
            for _ in range(2):
                restore_rng(saved_rng, acc); model.zero_grad(set_to_none=True)
                loss = model(observation(samples[rank], device), samples[rank]['actions'].unsqueeze(0).to(device)).mean()
                loss.backward(); acc.synchronize()
            if reference is None:
                reference_loss = loss.item()
                reference = {n: p.grad.detach().clone() for n, p in named.items() if p.grad is not None}
                emit('bucket_reference', {'loss': reference_loss, 'bucket_MiB': bucket})
            else:
                metrics = full_compare(reference)
                metrics.update(loss_abs_diff=abs(loss.item()-reference_loss), bucket_MiB=bucket)
                emit(f'bucket_numeric_{bucket}', metrics)
        del reference
        model.zero_grad(set_to_none=True); gc.collect(); acc.empty_cache()
        torch.use_deterministic_algorithms(False)

        settings = dict(lr=5e-5, betas=(.9, .95), eps=1e-8, weight_decay=1e-10)
        optim = torch.optim.AdamW(model.parameters(), **settings)
        optimizer_kind = 'standard'
        total_steps = 0
        rows = []
        probe_names = ['action_out_proj.weight', 'time_mlp_in.weight',
                       'paligemma_with_expert.paligemma.model.language_model.layers.0.self_attn.q_proj.weight']

        def update(label, warmup):
            nonlocal total_steps
            sample = samples[(total_steps*4+rank) % len(samples)]
            lr = 5e-5/10001+(5e-5-5e-5/10001)*total_steps/10000
            for group in optim.param_groups: group['lr'] = lr
            dist.barrier(); acc.synchronize(); acc.reset_peak_memory_stats()
            started = time.perf_counter()
            optim.zero_grad(set_to_none=True)
            loss = model(observation(sample, device), sample['actions'].unsqueeze(0).to(device)).mean()
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            acc.synchronize()
            elapsed = time.perf_counter()-started
            row = {'step': total_steps, 'label': label, 'warmup': warmup, 'seconds': elapsed,
                   'loss': loss.item(), 'grad_norm': norm.item(), 'lr': lr,
                   'peak_GiB': acc.max_memory_allocated()/2**30, 'sample': sample['id']}
            assert math.isfinite(row['loss']) and math.isfinite(row['grad_norm'])
            total_steps += 1; rows.append(row)
            return row

        def optimizer_numeric_probe():
            # Actual weights and clipped gradients, original dtypes; bounded representative slices.
            used = [(n, p) for n, p in named.items() if p.grad is not None]
            copies = [[torch.nn.Parameter(p.detach().flatten()[:4096].clone()) for n, p in used] for _ in range(2)]
            opts = [torch.optim.AdamW(copies[0], **settings),
                    torch.optim.AdamW(copies[1], fused=True, **settings)]
            for params in copies:
                for clone, (_, original) in zip(params, used):
                    clone.grad = original.grad.detach().flatten()[:4096].clone()
            checks = []
            for i in range(3):
                for opt in opts: opt.step()
                acc.synchronize()
                check = {'step': i+1, 'probe_lr': settings['lr']}
                for key in ['parameter', 'exp_avg', 'exp_avg_sq']:
                    vectors = []
                    for params, opt in zip(copies, opts):
                        tensors = [p if key == 'parameter' else opt.state[p][key] for p in params]
                        vectors.append(torch.cat([v.detach().float().flatten() for v in tensors]).cpu().double())
                    a, b = vectors
                    check[key] = {'elements': a.numel(), 'max_abs': (a-b).abs().max().item(),
                                  'relative_l2': ((a-b).norm()/a.norm().clamp_min(1e-30)).item(),
                                  'strict_match': bool(torch.allclose(a, b, atol=1e-7, rtol=1e-5)),
                                  'finite': bool(torch.isfinite(b).all())}
                checks.append(check)
            emit('fused_optimizer_numerics', {'parameter_tensors': len(used), 'checks': checks,
                 'scope': 'first 4096 elements per active tensor, real weights and clipped gradients; zero moments; three repeated identical-gradient updates at peak LR; not full-model convergence'})
            del opts, copies
            gc.collect(); acc.empty_cache()

        variants = [('baseline25', False, 25, False), ('layer25', True, 25, False),
                    ('layer_fused25', True, 25, True), ('layer50', True, 50, False),
                    ('layer100', True, 100, False), ('layer25_repeat', True, 25, False),
                    ('baseline25_repeat', False, 25, False)]
        summaries = {}
        for label, candidate, bucket, fused in variants:
            set_candidate(candidate)
            rewrap(bucket)
            next_kind = 'fused' if fused else 'standard'
            if optimizer_kind != next_kind:
                # State tensors retain their original dtype; step tensor placement follows the destination implementation.
                state = optim.state_dict()
                for group in state['param_groups']:
                    group['fused'] = True if fused else None
                    group['foreach'] = None
                if not fused:
                    for entry in state['state'].values():
                        if isinstance(entry.get('step'), torch.Tensor): entry['step'] = entry['step'].cpu()
                new_optim = torch.optim.AdamW(model.parameters(), fused=True, **settings) if fused else torch.optim.AdamW(model.parameters(), **settings)
                new_optim.load_state_dict(state)
                optim = new_optim; optimizer_kind = next_kind
                assert bool(optim.param_groups[0].get('fused')) == fused
                del state, new_optim
                gc.collect(); acc.empty_cache()
            before = {n: named[n].detach().flatten()[:4096].float().clone() for n in probe_names}
            for _ in range(3): update(label, True)
            results = [update(label, False) for _ in range(8)]
            changed = {n: int(torch.count_nonzero(named[n].detach().flatten()[:4096].float()-before[n]).item()) for n in probe_names}
            assert any(changed.values())
            all_finite = torch.stack([torch.isfinite(v).all() for s in optim.state.values()
                                      for k, v in s.items() if k in {'exp_avg', 'exp_avg_sq'}]).all().item()
            assert all_finite
            probe = torch.cat([named[n].grad.flatten()[:128].float() for n in probe_names])
            gathered = [torch.empty_like(probe) for _ in range(4)]
            dist.all_gather(gathered, probe)
            synchronization_error = max((probe-v).abs().max().item() for v in gathered)
            assert synchronization_error == 0
            ddp_data = model._get_ddp_logging_data()
            summary = {'rows': results, 'median_seconds': statistics.median(r['seconds'] for r in results),
                       'candidate': candidate, 'bucket_MiB': bucket, 'optimizer': optimizer_kind,
                       'optimizer_state_tensors': len(optim.state), 'all_moments_finite': all_finite,
                       'optimizer_fused_flag': optim.param_groups[0].get('fused'),
                       'optimizer_step_devices': sorted({str(s['step'].device) for s in optim.state.values()}),
                       'sampled_parameter_changes': changed, 'sampled_cross_rank_gradient_max_abs': synchronization_error,
                       'ddp': ddp_data}
            emit(label, summary)
            summaries[label] = summary
            if label == 'baseline25': optimizer_numeric_probe()
        emit('complete', {'status': 'benchmark_complete', 'updates': total_steps,
                          'variants': list(summaries), 'candidate_not_automatically_promoted': True,
                          'no_checkpoint_written': True, 'no_inference': True})
    dist.barrier(); dist.destroy_process_group()


if __name__ == '__main__': main()
