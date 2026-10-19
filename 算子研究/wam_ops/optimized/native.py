"""完整原生block；保留原模型BF16计算与FP32调制/Norm权重。"""
from functools import lru_cache
import importlib.util
from pathlib import Path
import weakref

import torch

TORCH_COMMIT = '0fabc3ba44823f257e70ce397d989c8de5e362c1'


@lru_cache(None)
def extension():
    if torch.version.git_version != TORCH_COMMIT:
        raise RuntimeError('native block requires the pinned Torch version')
    files = list(Path(__file__).parent.glob('wam_optimized_block_cuda*.so'))
    if len(files) != 1:
        raise RuntimeError('Build optimized/build_subgraph.py first')
    spec = importlib.util.spec_from_file_location('wam_optimized_block_cuda', files[0])
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class NativeBlock:
    """Bind the original parameter tensors; forward contains no Torch math ops."""
    def __init__(self, block, fused_ffn1=False, mode="sequential"):
        if mode not in ("sequential", "tiled"):
            raise ValueError("mode must be sequential or tiled")
        self.mode = mode
        self.block = weakref.ref(block)
        self.fused_ffn1 = fused_ffn1
        self.weights = [block.scale_shift_table]
        if hasattr(block.norm2, 'weight'):
            self.weights += [block.norm2.weight, block.norm2.bias]
        else:
            empty = block.scale_shift_table.new_empty(0)
            self.weights += [empty, empty]
        for a in (block.attn1, block.attn2):
            for m in (a.to_q, a.to_k, a.to_v, a.to_out[0]):
                self.weights += [m.weight, m.bias]
            self.weights += [a.norm_q.weight, a.norm_k.weight]
        for m in (block.ffn.net[0].proj, block.ffn.net[2]):
            self.weights += [m.weight, m.bias]
        # Read-only aliases preserve the original Parameters and grad flags.
        # Canonical inference uses no_grad; it need not mutate requires_grad.
        self.weights = [value.detach() for value in self.weights]
        self.eps = [block.norm1.eps, getattr(block.norm2, 'eps', block.norm1.eps),
                    block.norm3.eps, block.attn1.norm_q.eps, block.attn1.norm_k.eps,
                    block.attn2.norm_q.eps, block.attn2.norm_k.eps]

    def __call__(self, hidden_states, encoder_hidden_states, temb, rotary_emb,
                 update_cache=0, cache_name='pos', candidate_cache_rows=None, debug=False):
        if torch.is_grad_enabled():
            raise RuntimeError('NativeBlock is inference-only; use no_grad/inference_mode')
        block = self.block()
        cache = block.attn1.attn_caches.get(cache_name)
        values = ([] if cache is None else
                  [cache[k] for k in ('k', 'v', 'id', 'mask', 'is_pred')])
        outputs = extension().forward(self.weights, hidden_states, encoder_hidden_states,
            temb, rotary_emb, values, candidate_cache_rows, update_cache, self.eps, debug,
            self.fused_ffn1, self.mode == "tiled")
        return outputs if debug else outputs['output']
