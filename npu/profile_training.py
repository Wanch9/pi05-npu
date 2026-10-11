"""Bounded four-NPU training profile; same model, precision, data and global batch."""
import argparse
from contextlib import nullcontext
from datetime import timedelta
import hashlib
import json
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
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--weights', type=Path, required=True)
    parser.add_argument('--data', type=Path, required=True)
    args=parser.parse_args()
    rank=int(os.environ['RANK']);assert int(os.environ['WORLD_SIZE'])==4
    device=local_device();acc=torch.npu
    dist.init_process_group('hccl',timeout=timedelta(seconds=600))
    if rank==0:args.output.mkdir(parents=True,exist_ok=False)
    dist.barrier()
    def emit(event, data):
        (args.output/f'rank-{rank}-{event}.json').write_text(json.dumps(data,indent=2))
        print(json.dumps({'rank':rank,'event':event,**data}),flush=True)
    torch.set_num_threads(4)
    torch.manual_seed(42);np.random.seed(42);random.seed(42);acc.manual_seed_all(42)
    cfg=SimpleNamespace(pi05=True,paligemma_variant='gemma_2b',action_expert_variant='gemma_300m',
                        dtype='bfloat16',action_dim=32,action_horizon=10,max_token_len=200,
                        discrete_state_input=False,pytorch_compile_mode=None)
    emit('identity',{'pid':os.getpid(),'device':str(device),'world_size':4,'global_batch':4,
         'visible_devices':os.environ.get('ASCEND_RT_VISIBLE_DEVICES'),'config':vars(cfg),
         'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
         'scope':'training only; eight cached samples reused; no checkpoint or inference',
         'profiler':'rank 0 CPU+NPU Level1, one warmed step; separate from timing'})
    started=time.perf_counter()
    module=PI0Pytorch(cfg)
    safetensors.torch.load_model(module,str(args.weights),strict=True,device='cpu')
    module.to(device);module.gradient_checkpointing_enable();module.train()
    model=DDP(module,device_ids=[device.index],find_unused_parameters=True,gradient_as_bucket_view=True)
    optim=torch.optim.AdamW(model.parameters(),lr=5e-5,betas=(.9,.95),eps=1e-8,weight_decay=1e-10)
    samples=torch.load(args.data / 'samples.pt',weights_only=False)
    torch.manual_seed(42+rank);np.random.seed(42+rank);random.seed(42+rank);acc.manual_seed_all(42+rank)
    named=dict(module.named_parameters())
    names=['action_out_proj.weight','time_mlp_in.weight',
           'paligemma_with_expert.gemma_expert.model.layers.0.input_layernorm.dense.weight',
           'paligemma_with_expert.gemma_expert.model.layers.0.self_attn.q_proj.weight',
           'paligemma_with_expert.paligemma.model.language_model.layers.0.self_attn.q_proj.weight',
           'paligemma_with_expert.paligemma.model.language_model.layers.0.input_layernorm.weight',
           'paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.0.self_attn.q_proj.weight',
           'paligemma_with_expert.paligemma.model.vision_tower.vision_model.embeddings.patch_embedding.weight']
    original_checkpoint=module._apply_checkpoint
    def layer_only(func,*values,**kwargs):
        # Retain per-transformer-layer and image checkpointing; omit the nested whole-stack wrapper.
        if func.__name__=='forward_func':return func(*values,**kwargs)
        return original_checkpoint(func,*values,**kwargs)
    def mode(candidate):module._apply_checkpoint=layer_only if candidate else original_checkpoint
    acc.synchronize();dist.barrier()
    emit('loaded',{'seconds':time.perf_counter()-started,'parameters':sum(p.numel() for p in module.parameters()),
                   'strict_load':True,'trainable':sum(p.numel() for p in module.parameters() if p.requires_grad)})

    # Same weights, input and all RNG states; no optimizer update between the two graphs.
    saved_rng=rng_state(acc)
    reference=None
    equivalence={}
    comparisons={}
    # DDP may rebuild buckets after its first backward; stabilize before comparison.
    mode(False);optim.zero_grad(set_to_none=True)
    model(observation(samples[rank],device),samples[rank]['actions'].unsqueeze(0).to(device)).mean().backward()
    acc.synchronize()
    for comparison,candidate in [('reference',False),('candidate',True),('reference_repeat',False)]:
        mode(candidate);restore_rng(saved_rng,acc);optim.zero_grad(set_to_none=True)
        sample=samples[rank];obs=observation(sample,device);actions=sample['actions'].unsqueeze(0).to(device)
        loss=model(obs,actions).mean();loss.backward();acc.synchronize()
        used=[(n,p) for n,p in named.items() if p.grad is not None]
        pieces=[p.grad.flatten()[:64].float().cpu().double() for n,p in used]
        vectors=torch.cat(pieces)
        norms=torch.stack([p.grad.float().norm() for n,p in used]).cpu().double()
        assert torch.isfinite(vectors).all() and torch.isfinite(norms).all()
        if comparison=='reference':reference=(loss.item(),vectors,norms,pieces)
        else:
            equivalence={'loss_reference':reference[0],'loss_candidate':loss.item(),
                'loss_abs_diff':abs(loss.item()-reference[0]),'sampled_gradient_elements':vectors.numel(),
                'gradient_sample_max_abs':(vectors-reference[1]).abs().max().item(),
                'gradient_sample_relative_l2':((vectors-reference[1]).norm()/reference[1].norm()).item(),
                'all_gradient_tensor_norm_max_abs':(norms-reference[2]).abs().max().item()}
            equivalence['strict_sample_match']=bool(torch.allclose(vectors,reference[1],atol=1e-7,rtol=1e-5))
            differences=[{'name':n,'dtype':str(p.dtype),'sample_max_abs':(v-ref).abs().max().item(),
                          'gradient_norm_abs_diff':abs(norms[i].item()-reference[2][i].item())}
                         for i,((n,p),v,ref) in enumerate(zip(used,pieces,reference[3]))]
            equivalence['largest_differences']=sorted(differences,key=lambda x:x['sample_max_abs'],reverse=True)[:12]
            assert equivalence['loss_abs_diff']<=1e-7,equivalence
            comparisons[comparison]=equivalence
    comparisons['candidate_accepted_by_strict_check']=comparisons['candidate']['strict_sample_match']
    emit('equivalence',comparisons)
    del reference,vectors,norms
    restore_rng(saved_rng,acc)
    step=0
    rows=[]
    def update(label,audit=False,split=False):
        nonlocal step
        sample=samples[(step*4+rank)%len(samples)]
        lr=5e-5/10001+(5e-5-5e-5/10001)*step/10000
        for group in optim.param_groups:group['lr']=lr
        times={}
        def stage(name,fn):
            t=time.perf_counter()
            with torch.autograd.profiler.record_function('PI05/'+name):result=fn()
            if split:acc.synchronize()
            times[name]=time.perf_counter()-t
            return result
        dist.barrier();acc.synchronize();acc.reset_peak_memory_stats()
        t=time.perf_counter()
        optim.zero_grad(set_to_none=True)
        obs,actions=stage('input',lambda:(observation(sample,device),sample['actions'].unsqueeze(0).to(device)))
        before={n:named[n].detach().flatten()[:4096].float().clone() for n in names} if audit else None
        loss=stage('forward',lambda:model(obs,actions).mean())
        if audit:assert torch.isfinite(loss)
        stage('backward_hccl',loss.backward)
        def grad_audit():
            assert torch.stack([torch.isfinite(p.grad).all() for p in module.parameters() if p.grad is not None]).all()
            probe=torch.cat([named[n].grad.flatten()[:128].float() for n in names]);others=[torch.empty_like(probe) for _ in range(4)]
            dist.all_gather(others,probe)
            assert max((probe-v).abs().max().item() for v in others)==0
            return {n:(named[n].grad.float().norm().item(),torch.count_nonzero(named[n].grad).item()) for n in names}
        if audit:stage('gradient_audit',grad_audit)
        norm=stage('clip',lambda:torch.nn.utils.clip_grad_norm_(model.parameters(),1.0))
        stage('optimizer',optim.step)
        def moment_audit():
            checks=[torch.isfinite(v).all() for state in optim.state.values() for key,v in state.items() if key in {'exp_avg','exp_avg_sq'}]
            assert torch.stack(checks).all()
            changes=[torch.count_nonzero(named[n].detach().flatten()[:4096].float()-before[n]).item() for n in names]
            assert any(changes)
            assert {int(s['step'].item()) for s in optim.state.values()}=={step+1}
        if audit:stage('moment_parameter_audit',moment_audit)
        acc.synchronize();elapsed=time.perf_counter()-t
        row={'label':label,'step':step,'seconds':elapsed,'loss':loss.item(),'grad_norm':norm.item(),
             'peak_GiB':acc.max_memory_allocated()/2**30,'sample':sample['id'],'audit':audit,'split_sync':split}
        assert np.isfinite(row['loss']) and np.isfinite(row['grad_norm'])
        if split:row['stages_seconds']=times
        rows.append(row);step+=1
        return row
    def block(label,candidate,warmup=3,measured=5,audit=False):
        mode(candidate)
        for _ in range(warmup):update(label+'/warmup',audit)
        results=[update(label,audit) for _ in range(measured)]
        emit(label,{'rows':results,'median_seconds':statistics.median(r['seconds'] for r in results),
                    'peak_GiB':max(r['peak_GiB'] for r in results)})
    block('baseline',False)
    emit('baseline_breakdown',update('baseline_breakdown',split=True))
    block('layer_checkpoint_only',True)
    emit('candidate_breakdown',update('candidate_breakdown',split=True))
    block('baseline_repeat',False,warmup=1,measured=3)
    block('validation_overhead',False,warmup=0,measured=2,audit=True)
    emit('audit_breakdown',update('audit_breakdown',audit=True,split=True))
    emit('timing_complete',{'updates':step,'rows':rows})
    # Profiling overhead is excluded from all benchmark rows above.
    mode(False)
    ctx=torch_npu.profiler.profile(
        activities=[torch_npu.profiler.ProfilerActivity.CPU,torch_npu.profiler.ProfilerActivity.NPU],
        record_shapes=True,profile_memory=True,with_stack=False,
        experimental_config=torch_npu.profiler._ExperimentalConfig(profiler_level=torch_npu.profiler.ProfilerLevel.Level1),
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(str(args.output/'trace')),
    ) if rank==0 else nullcontext()
    with ctx as prof:
        update('profiler_capture')
        if rank==0:prof.step()
    dist.barrier()
    emit('complete',{'status':'profile_complete','candidate_strict_gradient_match':comparisons['candidate_accepted_by_strict_check'],
                     'updates':step,'optimizer_states':len(optim.state),
                     'no_checkpoint_written':True,'no_inference':True})
    dist.destroy_process_group()


if __name__=='__main__':main()
