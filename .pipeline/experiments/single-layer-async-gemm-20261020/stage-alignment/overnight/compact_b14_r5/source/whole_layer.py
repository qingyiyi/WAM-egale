"""单层Transformer的一次persistent launch。固定实验输入；请在inference_mode下调用。"""
from functools import lru_cache
import importlib.util
from pathlib import Path


@lru_cache(None)
def extension(compact_b14=False):
    name = "wam_whole_b14_cuda" if compact_b14 else "wam_whole_fa2_cuda"
    path = next(Path(__file__).parent.glob(name + "*.so"))
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class WholeLayerBlock:
    """绑定原参数和KV；数学计算都在csrc/whole_layer.cu中。"""
    production_admitted = False

    def __init__(self, block):
        # 只复用参数列表的绑定，不调用NativeBlock.forward。
        from .native import NativeBlock
        binding = NativeBlock(block)
        self.weights, self.eps = binding.weights, binding.eps
        self.block = block

    def __call__(self, hidden_states, encoder_hidden_states, temb, rotary_emb,
                 update_cache=0, cache_name="pos", candidate_cache_rows=None,
                 debug=False):
        cache = self.block.attn1.attn_caches.get(cache_name)
        values = [] if cache is None else [
            cache[key] for key in ("k", "v", "id", "mask", "is_pred")
        ]
        compact = (candidate_cache_rows is not None and
                   tuple(hidden_states.shape[:2]) == (14, 16))
        result = extension(compact).forward(
            self.weights, hidden_states, encoder_hidden_states, temb, rotary_emb,
            values, candidate_cache_rows, update_cache, self.eps,
        )
        # 中间量仅供外部核验；普通调用返回与hidden同形状的输出。
        return result if debug else result["output"]
