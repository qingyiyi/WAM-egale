"""一层、一次 cooperative launch 的研究入口。未通过原逐位门，禁止生产替换。"""
from functools import lru_cache
import importlib.util
from pathlib import Path

import torch


@lru_cache(None)
def extension():
    if torch.version.git_version != '0fabc3ba44823f257e70ce397d989c8de5e362c1':
        raise RuntimeError('整层扩展要求固定Torch2.9源码版本')
    files = list(Path(__file__).parent.glob('wam_whole_layer_cuda*.so'))
    if len(files) != 1:
        raise RuntimeError('先构建 optimized/build_whole_layer.py')
    spec = importlib.util.spec_from_file_location('wam_whole_layer_cuda', files[0])
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class WholeLayerBlock:
    """保留原参数别名、缓存与 public forward；默认不接入任何模型。"""
    production_admitted = False

    def __init__(self, block):
        # 只借用冻结参数绑定，不调用旧数学/DSO。
        from .native import NativeBlock
        binding = NativeBlock(block)
        self.weights, self.eps = binding.weights, binding.eps
        self.block = block

    def __call__(self, hidden_states, encoder_hidden_states, temb, rotary_emb,
                 update_cache=0, cache_name='pos', candidate_cache_rows=None,
                 debug=False, audit=True):
        if torch.is_grad_enabled():
            raise RuntimeError('WholeLayerBlock 仅支持 no_grad/inference_mode')
        cache = self.block.attn1.attn_caches.get(cache_name)
        values = ([] if cache is None else
                  [cache[k] for k in ('k', 'v', 'id', 'mask', 'is_pred')])
        result = extension().forward(self.weights, hidden_states, encoder_hidden_states,
            temb, rotary_emb, values, candidate_cache_rows, update_cache, self.eps, audit)
        return result if debug else result['output']
