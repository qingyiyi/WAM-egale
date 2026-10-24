"""Load exact block-0 weights and original class bodies; no full-model load."""
import ast
import math
from functools import partial
from pathlib import Path
from typing import Callable, ClassVar
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.attention.flex_attention import (
    _mask_mod_signature, BlockMask, create_block_mask, flex_attention, and_masks, or_masks)
from diffusers.models.attention import FeedForward
from diffusers.models.normalization import FP32LayerNorm
from einops import rearrange

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / 'assets/fixtures'

def load_block(device='cuda'):
    source = ROOT / 'source/model.py'
    tree = ast.parse(source.read_text())
    names = {'custom_sdpa', 'FlexAttnFunc', 'WanAttention', 'WanTransformerBlock'}
    nodes = [n for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name in names]
    namespace = dict(globals())
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), 'exec'), namespace)
    saved = torch.load(ROOT / 'assets/block0.pt', map_location='cpu', mmap=True, weights_only=False)
    c = saved['config']
    with torch.device('meta'):
        block = namespace['WanTransformerBlock'](
            dim=c['num_attention_heads'] * c['attention_head_dim'], ffn_dim=c['ffn_dim'],
            num_heads=c['num_attention_heads'], cross_attn_norm=c['cross_attn_norm'],
            eps=c['eps'], attn_mode='torch')
    block.load_state_dict(saved['state_dict'], strict=True, assign=True)
    for name, parameter in block.named_parameters():
        if name == 'scale_shift_table' or name.startswith('norm2.'):
            parameter.data = parameter.data.float()
    return block.to(device).eval().requires_grad_(False)
