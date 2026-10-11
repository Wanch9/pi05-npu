"""Bounded full-parameter PI0.5 update and exact-state resume on four accelerators."""
import argparse
from collections import Counter
from datetime import timedelta
import hashlib
import json
import os
from pathlib import Path
import random
import socket
import time
from types import SimpleNamespace

import numpy as np
import safetensors.torch
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from openpi.models_pytorch.device import accelerator, backend, local_device
from openpi.models_pytorch.pi0_pytorch import PI0Pytorch


def cpu_tree(value):
    if isinstance(value, torch.Tensor):return value.detach().cpu()
    if isinstance(value, dict):return {k:cpu_tree(v) for k,v in value.items()}
    if isinstance(value, list):return [cpu_tree(v) for v in value]
    if isinstance(value, tuple):return tuple(cpu_tree(v) for v in value)
    return value


def observation(sample, device):
    return SimpleNamespace(
        images={k:v.unsqueeze(0).to(device) for k,v in sample['images'].items()},
        image_masks={k:torch.tensor([k!='right_wrist_0_rgb'],device=device) for k in sample['images']},
        state=sample['state'].unsqueeze(0).to(device),
        tokenized_prompt=sample['tokenized_prompt'].unsqueeze(0).to(device),
        tokenized_prompt_mask=sample['tokenized_prompt_mask'].unsqueeze(0).to(device),
        token_ar_mask=None,
        token_loss_mask=None,
    )


def rng_state(acc):
    return {'torch':torch.get_rng_state(), 'numpy':np.random.get_state(), 'python':random.getstate(),
            'accelerator':acc.get_rng_state() if acc else None}


def restore_rng(state, acc):
    torch.set_rng_state(state['torch']);np.random.set_state(state['numpy']);random.setstate(state['python'])
    if acc:acc.set_rng_state(state['accelerator'])


def emit(out, rank, name, value):
    (out/f'rank-{rank}-{name}.json').write_text(json.dumps(value,indent=2))
    print(json.dumps({'rank':rank,'event':name,**value}),flush=True)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--weights',type=Path,required=True)
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--resume',type=Path)
    p.add_argument('--max-steps',type=int,required=True)
    p.add_argument('--inference',action='store_true')
    a=p.parse_args()
    rank=int(os.environ['RANK']);world=int(os.environ['WORLD_SIZE'])
    assert world==4, 'This validation contract requires exactly four ranks'
    assert a.max_steps in (1,2)
    device=local_device();acc=accelerator(device)
    dist.init_process_group(backend(device),timeout=timedelta(seconds=600))
    a.output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4)
    torch.manual_seed(42);np.random.seed(42);random.seed(42)
    if acc:acc.manual_seed_all(42)
    cfg=SimpleNamespace(pi05=True,paligemma_variant='gemma_2b',action_expert_variant='gemma_300m',
                        dtype='bfloat16',action_dim=32,action_horizon=10,max_token_len=200,
                        discrete_state_input=False,pytorch_compile_mode=None)
    emit(a.output,rank,'identity',{'host':socket.gethostname(),'pid':os.getpid(),'device':str(device),
         'local_rank':int(os.environ['LOCAL_RANK']),
         'world_size':world,'backend':dist.get_backend(),'visible_devices':os.environ.get('ASCEND_RT_VISIBLE_DEVICES'),
         'batch_per_rank':1,'global_batch':4,'accumulation':1,'config':vars(cfg),
         'precision':'upstream selected BF16 weights; norms/patch embedding/projections FP32; no autocast',
         'source_manifest_sha256':hashlib.sha256((Path(__file__).parent/'SOURCE_MANIFEST.json').read_bytes()).hexdigest(),
         'source_revision':'215abfb217dbac7d5f1273282331b9b1866c0479'})
    model=PI0Pytorch(cfg)
    loaded=None
    if a.resume:
        loaded=torch.load(a.resume,map_location='cpu',weights_only=False)
        assert loaded['world_size']==4 and loaded['config']==vars(cfg)
        model.load_state_dict(loaded['model'],strict=True)
        step=loaded['completed_steps']
        del loaded['model']
    else:
        safetensors.torch.load_model(model,str(a.weights),strict=True,device='cpu')
        step=0
    assert all(x.requires_grad for x in model.parameters())
    model.to(device);model.gradient_checkpointing_enable()
    model=DDP(model,device_ids=[device.index] if device.type!='cpu' else None,
              find_unused_parameters=True,gradient_as_bucket_view=True)
    optimizer=torch.optim.AdamW(model.parameters(),lr=5e-5,betas=(.9,.95),eps=1e-8,weight_decay=1e-10)
    if loaded:
        optimizer.load_state_dict(loaded['optimizer']);del loaded['optimizer']
        restore_rng(loaded['rng'][rank],acc)
        assert loaded['next_sample_index']==step*4
    else:
        torch.manual_seed(42+rank);np.random.seed(42+rank);random.seed(42+rank)
        if acc:acc.manual_seed_all(42+rank)
    samples=torch.load(a.data/'samples.pt',map_location='cpu',weights_only=False)
    data_identity={'receipt':json.loads((a.data/'READY.json').read_text()),
                   'normalization':json.loads((a.data/'norm_stats.json').read_text())}
    if loaded:
        assert loaded['data_identity']==data_identity, 'Resume dataset or normalization changed'
    names=[
        'action_out_proj.weight','time_mlp_in.weight',
        'paligemma_with_expert.gemma_expert.model.layers.0.input_layernorm.dense.weight',
        'paligemma_with_expert.gemma_expert.model.layers.0.self_attn.q_proj.weight',
        'paligemma_with_expert.paligemma.model.language_model.layers.0.self_attn.q_proj.weight',
        'paligemma_with_expert.paligemma.model.language_model.layers.0.input_layernorm.weight',
        'paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.0.self_attn.q_proj.weight',
        'paligemma_with_expert.paligemma.model.vision_tower.vision_model.embeddings.patch_embedding.weight',
    ]
    named=dict(model.module.named_parameters())
    assert set(names)<=set(named), set(names)-set(named)
    emit(a.output,rank,'loaded',{'strict_load':True,'completed_steps':step,
         'parameters':sum(x.numel() for x in model.parameters()),'trainable_parameters':sum(x.numel() for x in model.parameters() if x.requires_grad),
         'dtype_numel':dict(Counter({str(d):sum(x.numel() for x in model.parameters() if x.dtype==d) for d in {x.dtype for x in model.parameters()}})),
         'resume':bool(a.resume),'optimizer_states':len(optimizer.state)})
    while step<a.max_steps:
        assert step*4+rank<len(samples)
        sample=samples[step*4+rank]
        obs=observation(sample,device);actions=sample['actions'].unsqueeze(0).to(device)
        # Same first-step learning-rate schedule as upstream pi05_libero.
        lr=5e-5/10001+(5e-5-5e-5/10001)*step/10000
        for group in optimizer.param_groups:group['lr']=lr
        model.train();optimizer.zero_grad(set_to_none=True)
        before={name:named[name].detach().reshape(-1)[:4096].float().clone() for name in names}
        t=time.monotonic()
        loss=model(obs,actions).mean()
        if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss')
        loss.backward()
        grads=[v.grad for v in named.values() if v.grad is not None]
        finite=torch.stack([torch.isfinite(g).all() for g in grads]).all()
        if not finite:raise RuntimeError('Nonfinite gradients')
        # DDP has completed gradient averaging here; compare actual values across ranks.
        probe=torch.cat([named[n].grad.detach().reshape(-1)[:128].float() for n in names])
        gathered=[torch.empty_like(probe) for _ in range(world)]
        dist.all_gather(gathered,probe)
        sync_error=max((probe-x).abs().max().item() for x in gathered)
        assert sync_error==0, sync_error
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
        assert torch.isfinite(norm), 'Nonfinite gradient norm'
        grad_measures={n:{'norm':named[n].grad.float().norm().item(),'nonzero':int(torch.count_nonzero(named[n].grad).item())} for n in names}
        optimizer.step()
        if acc:acc.synchronize()
        moment_checks=[torch.isfinite(value).all() for state in optimizer.state.values()
                       for key,value in state.items() if key in {'exp_avg','exp_avg_sq'}]
        assert moment_checks and torch.stack(moment_checks).all(), 'Nonfinite optimizer moments'
        changes={}
        for n in names:
            delta=named[n].detach().reshape(-1)[:4096].float()-before[n]
            assert torch.isfinite(delta).all(), n
            changes[n]={'sampled_elements':delta.numel(),'changed':int(torch.count_nonzero(delta).item()),'max_abs':delta.abs().max().item(),**grad_measures[n]}
        assert any(x['changed']>0 for x in changes.values()), 'No measured parameter update'
        step+=1
        optim_steps=sorted({int(s['step'].item()) for s in optimizer.state.values() if 'step' in s})
        assert optim_steps==[step]
        emit(a.output,rank,f'update-{step}',{'completed_steps':step,'sample':sample['id'],'loss':loss.item(),
             'gradient_norm_before_clip':norm.item(),'gradients_finite':True,'parameters_with_grad':len(grads),
             'sampled_gradient_sync_max_abs':sync_error,'parameter_changes':changes,'optimizer_steps':optim_steps,
             'optimizer_moments_finite':True,
             'lr':lr,'seconds':time.monotonic()-t,'peak_memory_bytes':acc.max_memory_allocated() if acc else None})
    # Persist each rank's RNG; collecting CPU RNG records avoids HCCL object collectives.
    torch.save(rng_state(acc),a.output/f'rng-{rank}.pt')
    dist.barrier()
    if rank==0:
        checkpoint={'model':cpu_tree(model.module.state_dict()),'optimizer':cpu_tree(optimizer.state_dict()),
                    'completed_steps':step,'next_sample_index':step*4,'world_size':world,'config':vars(cfg),
                    'data_identity':data_identity,
                    'rng':[torch.load(a.output/f'rng-{i}.pt',weights_only=False) for i in range(world)],
                    'schedule':{'warmup_steps':10000,'peak_lr':5e-5,'decay_steps':1000000,'decay_lr':5e-5}}
        tmp=a.output/'checkpoint.pt.tmp';torch.save(checkpoint,tmp);tmp.rename(a.output/'checkpoint.pt')
        del checkpoint
        emit(a.output,rank,'saved',{'completed_steps':step,'path':str(a.output/'checkpoint.pt'),'bytes':(a.output/'checkpoint.pt').stat().st_size})
    dist.barrier()
    if a.inference:
        model.eval()
        with torch.no_grad():result=model.module.sample_actions(device,observation(samples[rank],device),num_steps=10)
        assert result.shape==(1,10,32) and torch.isfinite(result).all()
        torch.save(result.cpu(),a.output/f'rank-{rank}-actions.pt')
        emit(a.output,rank,'inference',{'shape':list(result.shape),'finite':True,'steps':10,'abs_max':result.abs().max().item()})
    emit(a.output,rank,'complete',{'completed_steps':step,'status':'passed','amp_skipped_steps':0})
    dist.barrier();dist.destroy_process_group()


if __name__=='__main__':main()
