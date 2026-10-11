"""Materialize eight real LIBERO windows from the existing OBS-backed cache."""
import argparse
import io
import json
from pathlib import Path

import av
import numpy as np
import pyarrow.parquet as pq
import sentencepiece
import torch

from openpi.shared.image_tools import resize_with_pad_torch


def frame(path, index):
    with av.open(str(path)) as container:
        for i, image in enumerate(container.decode(video=0)):
            if i == index:
                return image.to_ndarray(format="rgb24")
    raise RuntimeError(f"Missing frame {index}: {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--tokenizer', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--source-label', default='existing local LIBERO cache')
    args = parser.parse_args()
    args.output.mkdir(exist_ok=False)
    tasks = {x['task_index']: x['task'] for x in map(json.loads, (args.root/'meta/tasks.jsonl').read_text().splitlines())}
    tokenizer = sentencepiece.SentencePieceProcessor(model_file=str(args.tokenizer))
    tables = [pq.read_table(args.root/f'data/chunk-000/episode_{i:06d}.parquet').to_pydict() for i in range(8)]
    stats = {}
    for key in ['action', 'observation.state']:
        values = np.concatenate([np.asarray(t[key], dtype=np.float32) for t in tables])
        stats[key] = {q: np.quantile(values, v, axis=0).astype(np.float32) for q, v in [('q01', .01), ('q99', .99)]}
    samples = []
    for episode, table in enumerate(tables):
        index = 16
        assert len(table['action']) >= index+10
        prompt = tasks[int(table['task_index'][index])]
        # Matches PaligemmaTokenizer without discrete state, as in pi05_libero.
        cleaned = prompt.strip().replace('_', ' ').replace('\n', ' ')
        tokens = tokenizer.encode(cleaned, add_bos=True) + tokenizer.encode('\n')
        assert len(tokens) <= 200
        images = {}
        for source, target in [('image', 'base_0_rgb'), ('wrist_image', 'left_wrist_0_rgb')]:
            path = args.root/f'videos/chunk-000/observation.images.{source}/episode_{episode:06d}.mp4'
            rgb = frame(path, index)
            # Same uint8 -> [-1,1] and bilinear letterbox contract as OpenPI.
            tensor = torch.from_numpy(rgb.copy()).float()/127.5-1.0
            resized = resize_with_pad_torch(tensor, 224, 224)
            assert resized.shape == (224, 224, 3), resized.shape
            images[target] = resized.permute(2,0,1).contiguous()
        images['right_wrist_0_rgb'] = torch.full_like(images['base_0_rgb'], -1)
        state = np.asarray(table['observation.state'][index], dtype=np.float32)
        actions = np.asarray(table['action'][index:index+10], dtype=np.float32)
        for key, values in [('observation.state', state), ('action', actions)]:
            s = stats[key]
            values[:] = (values-s['q01'])/(s['q99']-s['q01']+1e-6)*2-1
        sample = {
            'id': f'libero_spatial/episode_{episode:06d}/frame_{index}',
            'prompt': prompt, 'images': images,
            'state': torch.from_numpy(np.pad(state, (0,32-len(state)))),
            'actions': torch.from_numpy(np.pad(actions, ((0,0),(0,32-actions.shape[-1])))),
            'tokenized_prompt': torch.tensor(tokens+[0]*(200-len(tokens)), dtype=torch.long),
            'tokenized_prompt_mask': torch.tensor([True]*len(tokens)+[False]*(200-len(tokens))),
        }
        for key in ['state','actions']: assert torch.isfinite(sample[key]).all()
        samples.append(sample)
    torch.save(samples, args.output/'samples.pt')
    (args.output/'norm_stats.json').write_text(json.dumps({k:{q:v.tolist() for q,v in s.items()} for k,s in stats.items()},indent=2))
    receipt = {'source':str(args.root), 'obs_source':args.source_label,
               'samples':[x['id'] for x in samples], 'distinct_episodes':8, 'window':10,
               'state_dim':8, 'action_dim':7, 'padded_dim':32, 'max_token_len':200,
               'normalization':'q01/q99 computed only on these eight training episodes',
               'discrete_state_input':False, 'extra_delta_transform':False,
               'source_read_only':True, 'dataset_downloaded':False}
    (args.output/'READY.json').write_text(json.dumps(receipt,indent=2))
    print(json.dumps(receipt),flush=True)


if __name__ == '__main__':main()
