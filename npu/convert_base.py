"""Use upstream conversion, rejecting missing keys instead of silent random weights."""
import argparse
import importlib.util
import json
from pathlib import Path

from openpi.models.pi0_config import Pi0Config
from openpi.models_pytorch.pi0_pytorch import PI0Pytorch


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--source',required=True)
    p.add_argument('--output',required=True)
    a=p.parse_args()
    path=Path(__file__).resolve().parents[1]/'examples/convert_jax_model_to_pytorch.py'
    spec=importlib.util.spec_from_file_location('upstream_conversion',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    original=PI0Pytorch.load_state_dict
    def strict(self, values, strict=True, **kwargs):
        # Upstream conversion may omit tied weights and an unused expert lm_head.
        result=original(self,values,strict=False,**kwargs)
        allowed={'paligemma_with_expert.gemma_expert.lm_head.weight',
                 'paligemma_with_expert.paligemma.lm_head.weight'}
        assert not result.unexpected_keys, result.unexpected_keys
        assert set(result.missing_keys)<=allowed, result.missing_keys
        print(json.dumps({'conversion_missing_keys':result.missing_keys,'unexpected_keys':result.unexpected_keys,
                          'note':'expert lm_head is unused by PI0 flow head; PaliGemma lm_head is tied to embeddings'}),flush=True)
        return result
    PI0Pytorch.load_state_dict=strict
    cfg=Pi0Config(pi05=True,action_horizon=10,discrete_state_input=False,pytorch_compile_mode=None,dtype='float32')
    module.convert_pi0_checkpoint(a.source,'float32',a.output,cfg)


if __name__=='__main__':main()
