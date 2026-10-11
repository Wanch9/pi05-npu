"""Real-image CPU/NPU SigLIP and sinusoidal-path comparisons."""
import argparse
import json
from pathlib import Path

import safetensors
import torch
from transformers import SiglipVisionModel
from transformers.models.auto import CONFIG_MAPPING

from openpi.models_pytorch.device import local_device
from openpi.models_pytorch.pi0_pytorch import create_sinusoidal_pos_embedding


def metrics(reference, candidate):
    # Accumulate on CPU in float64: long FP32 reductions can report cosine > 1.
    x=reference.cpu().double().reshape(-1);y=candidate.cpu().double().reshape(-1)
    assert torch.isfinite(x).all(), 'Nonfinite CPU reference'
    delta=y-x
    return {'finite':bool(torch.isfinite(y).all()),'max_abs':delta.abs().max().item(),
            'mean_abs':delta.abs().mean().item(),'rmse':delta.square().mean().sqrt().item(),
            'reference_rms':x.square().mean().sqrt().item(),
            'relative_l2':(delta.norm()/x.norm().clamp_min(1e-12)).item(),
            'cosine':torch.nn.functional.cosine_similarity(x,y,dim=0).item()}


def main():
    p=argparse.ArgumentParser();p.add_argument('--weights',required=True);p.add_argument('--data',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();device=local_device();torch.set_num_threads(16)
    samples=torch.load(Path(a.data)/'samples.pt',weights_only=False)
    time=torch.tensor([.001,.3,.999])
    reference=create_sinusoidal_pos_embedding(time,1024,.004,4.,torch.device('cpu')).float()
    actual=create_sinusoidal_pos_embedding(time.to(device),1024,.004,4.,device).cpu()
    assert torch.equal(reference,actual), 'Sinusoidal CPU fallback changed the reference values'
    config=CONFIG_MAPPING['paligemma']().vision_config
    config.intermediate_size=4304;config.projection_dim=2048;config.projector_hidden_act='gelu_fast'
    # Matches the actual PI0Pytorch vision tower in this pinned runtime.
    config._attn_implementation='sdpa'
    model=SiglipVisionModel(config).eval().float()
    prefix='paligemma_with_expert.paligemma.model.vision_tower.'
    with safetensors.safe_open(a.weights,framework='pt',device='cpu') as f:
        state={k[len(prefix):]:f.get_tensor(k) for k in f.keys() if k.startswith(prefix)}
    model.load_state_dict(state,strict=True);del state
    inputs=torch.stack([samples[0]['images']['base_0_rgb'],samples[0]['images']['left_wrist_0_rgb']])
    result={'samples':[samples[0]['id']],'views':['base_0_rgb','left_wrist_0_rgb'],
            'batch_per_encoder_call':1,
            'sinusoidal_fp32_exact':True,'weights':a.weights,'reference':'CPU FP32 SDPA; identical real preprocessed images',
            'attention_implementation':model.config._attn_implementation}
    result['statistics_dtype']='CPU float64'
    with torch.no_grad():
        ref=torch.cat([model(x.unsqueeze(0)).last_hidden_state for x in inputs])
        model=model.to(device)
        out=torch.cat([model(x.unsqueeze(0).to(device)).last_hidden_state.cpu() for x in inputs])
        result['npu_fp32']=metrics(ref,out)
        model=model.bfloat16()
        for name,param in model.named_parameters():
            if any(x in name for x in ['embeddings.patch_embedding.weight','embeddings.patch_embedding.bias','embeddings.position_embedding.weight']):param.data=param.data.float()
        low=torch.cat([model(x.unsqueeze(0).to(device)).last_hidden_state.cpu() for x in inputs])
        result['npu_selected_bf16']=metrics(ref,low)
    # This is a measurement report, not an uncalibrated task-quality claim.
    assert result['npu_fp32']['finite'] and result['npu_selected_bf16']['finite']
    Path(a.output).write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)


if __name__=='__main__':main()
