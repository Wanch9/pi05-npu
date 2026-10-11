"""Four-card regression: native model switch versus the historical experiment.

Uses the same setup_ddp helper as scripts/train_pytorch.py, compares every active
gradient, then performs one real update. No checkpoint, downloads, or inference.
This does not exercise the regular LeRobot/W&B data pipeline.
"""
import argparse
import json
import os
from pathlib import Path
import random
from types import SimpleNamespace

import numpy as np
import safetensors.torch
import torch
import torch.distributed as dist

from openpi.models_pytorch.device import setup_ddp
from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
from npu.validate_four_cards import observation, restore_rng, rng_state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weights', required=True)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert int(os.environ['WORLD_SIZE']) == 4
    _, rank, device = setup_ddp(npu_training_optimization=True)
    if rank == 0:
        args.output.mkdir(parents=True, exist_ok=False)
    dist.barrier()
    torch.set_num_threads(4)
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    torch.npu.manual_seed_all(42)
    cfg = SimpleNamespace(pi05=True, paligemma_variant='gemma_2b', action_expert_variant='gemma_300m',
                          dtype='bfloat16', action_dim=32, action_horizon=10, max_token_len=200,
                          discrete_state_input=False, pytorch_compile_mode=None)
    module = PI0Pytorch(cfg)
    safetensors.torch.load_model(module, args.weights, strict=True, device='cpu')
    module.to(device)
    module.gradient_checkpointing_enable()
    module.train()
    model = torch.nn.parallel.DistributedDataParallel(
        module, device_ids=[device.index], find_unused_parameters=True,
        gradient_as_bucket_view=True, bucket_cap_mb=25,
    )
    samples = torch.load(args.data / 'samples.pt', weights_only=False)
    assert len(samples) == 8
    sample = samples[rank]
    torch.manual_seed(42 + rank)
    np.random.seed(42 + rank)
    random.seed(42 + rank)
    torch.npu.manual_seed_all(42 + rank)
    state = rng_state(torch.npu)
    original = module._apply_checkpoint

    def legacy(func, *values, **kwargs):
        if func.__name__ == 'forward_func':
            return func(*values, **kwargs)
        return original(func, *values, **kwargs)

    def backward():
        restore_rng(state, torch.npu)
        model.zero_grad(set_to_none=True)
        loss = model(observation(sample, device), sample['actions'].unsqueeze(0).to(device)).mean()
        loss.backward()
        torch.npu.synchronize()
        assert torch.isfinite(loss)
        return loss.item()

    module._apply_checkpoint = legacy
    backward()  # Stabilize DDP buckets, without an optimizer update.
    reference_loss = backward()
    reference = {n: p.grad.detach().clone() for n, p in module.named_parameters() if p.grad is not None}
    module._apply_checkpoint = original
    module.gradient_checkpointing_enable(checkpoint_whole_stack=False)
    assert module.paligemma_with_expert.paligemma.language_model.gradient_checkpointing
    assert module.paligemma_with_expert.paligemma.vision_tower.gradient_checkpointing
    assert module.paligemma_with_expert.gemma_expert.model.gradient_checkpointing
    candidate_loss = backward()
    counts, differences, elements = [], [], 0
    for name, p in module.named_parameters():
        if p.grad is None:
            assert name not in reference
            continue
        ref, actual = reference[name].flatten(), p.grad.detach().flatten()
        elements += actual.numel()
        for offset in range(0, actual.numel(), 16 * 1024 * 1024):
            a = ref[offset:offset + 16 * 1024 * 1024].float()
            b = actual[offset:offset + 16 * 1024 * 1024].float()
            delta = (a - b).abs()
            differences.append(delta.max())
            counts.append((~torch.isfinite(delta) | (delta > 1e-7 + 1e-5 * a.abs())).sum())
    max_abs = torch.stack(differences).max().item()
    outside = torch.stack(counts).sum().item()
    tensors = len(reference)
    assert outside == 0 and candidate_loss == reference_loss
    del reference, counts, differences, ref, actual, a, b, delta
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, betas=(0.9, 0.95), eps=1e-8, weight_decay=1e-10)
    before = module.action_out_proj.weight.detach().clone()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    assert torch.isfinite(norm)
    optimizer.step()
    torch.npu.synchronize()
    change = (module.action_out_proj.weight.detach() - before).abs().max().item()
    assert change > 0
    assert all(torch.isfinite(v['exp_avg']).all() and torch.isfinite(v['exp_avg_sq']).all()
               for v in optimizer.state.values())
    report = dict(status='passed', rank=rank, world_size=4, global_batch=4,
                  HCCL_DETERMINISTIC=os.environ['HCCL_DETERMINISTIC'],
                  checkpoint_whole_stack=module.checkpoint_whole_stack, bucket_cap_mb=25,
                  parameters=sum(p.numel() for p in module.parameters()), gradient_elements=elements,
                  gradient_tensors=tensors, gradient_max_abs=max_abs, outside_tolerance=outside,
                  atol=1e-7, rtol=1e-5, reference_loss=reference_loss, candidate_loss=candidate_loss,
                  optimizer_updates=1, optimizer_states=len(optimizer.state), action_out_max_change=change,
                  scope='native model switch vs historical experimental switch; shared regular-entry DDP setup; cached samples')
    (args.output / f'rank-{rank}.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
