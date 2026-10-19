"""只从已有checkpoint读取一个真实block，避免反复加载完整5B模型。"""
import json
from pathlib import Path
import sys

import torch
from safetensors import safe_open

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / '投机采样/results/runtime/p0-operator-torch-alignment-v2-20261018-r3/process1'
CHECKPOINT = ROOT / 'models/lingbot-va-posttrain-libero-long-modelscope-20260922/transformer'


def load_block():
    sys.path.insert(0, str(ROOT / '投机采样/wam-p0-worktree'))
    from wan_va.modules.model import WanTransformerBlock
    config = json.loads((CHECKPOINT / 'config.json').read_text())
    index = json.loads((CHECKPOINT / 'diffusion_pytorch_model.safetensors.index.json').read_text())['weight_map']
    selected = {key: shard for key, shard in index.items() if key.startswith('blocks.0.')}
    state = {}
    for shard in sorted(set(selected.values())):
        with safe_open(CHECKPOINT / shard, framework='pt', device='cpu') as source:
            for key, filename in selected.items():
                if filename == shard:
                    state[key.removeprefix('blocks.0.')] = source.get_tensor(key)
    with torch.device('meta'):
        block = WanTransformerBlock(dim=config['num_attention_heads'] * config['attention_head_dim'],
            ffn_dim=config['ffn_dim'], num_heads=config['num_attention_heads'],
            cross_attn_norm=config['cross_attn_norm'], eps=config['eps'], attn_mode='torch')
    block.load_state_dict(state, strict=True, assign=True)
    return block.to('cuda').eval().requires_grad_(False)
