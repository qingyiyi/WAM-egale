"""单层Transformer的一次persistent launch。固定实验输入；请在inference_mode下调用。"""
from functools import lru_cache
import importlib.util
from pathlib import Path


@lru_cache(None)
def extension():
    path = next(Path(__file__).parent.glob("wam_whole_fa2_cuda*.so"))
    spec = importlib.util.spec_from_file_location("wam_whole_fa2_cuda", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class WholeLayerBlock:
    """绑定原参数和KV；数学计算都在csrc/whole_layer.cu中。"""
    production_admitted = False

    def __init__(self, block):
        self.block = block
        # 与CUDA的w[]顺序一致：调制表、Norm2、self/cross投影与RMS、FFN。
        empty = block.scale_shift_table.new_empty(0)
        weights = [block.scale_shift_table, getattr(block.norm2, 'weight', empty),
                   getattr(block.norm2, 'bias', empty)]
        for attention in (block.attn1, block.attn2):
            for projection in (attention.to_q, attention.to_k, attention.to_v,
                               attention.to_out[0]):
                weights += [projection.weight, projection.bias]
            weights += [attention.norm_q.weight, attention.norm_k.weight]
        for projection in (block.ffn.net[0].proj, block.ffn.net[2]):
            weights += [projection.weight, projection.bias]
        self.weights = [value.detach() for value in weights]
        self.eps = [block.norm1.eps, getattr(block.norm2, 'eps', block.norm1.eps),
                    block.norm3.eps, block.attn1.norm_q.eps, block.attn1.norm_k.eps,
                    block.attn2.norm_q.eps, block.attn2.norm_k.eps]

    def __call__(self, hidden_states, encoder_hidden_states, temb, rotary_emb,
                 update_cache=0, cache_name="pos", candidate_cache_rows=None,
                 debug=False, ready_tiles=False):
        cache = self.block.attn1.attn_caches.get(cache_name)
        values = [] if cache is None else [
            cache[key] for key in ("k", "v", "id", "mask", "is_pred")
        ]
        # 旧DSO仍是已验证默认路径；新调度须显式启用，不借用旧精度/速度结论。
        options = {"ready_tiles": True} if ready_tiles else {}
        result = extension().forward(
            self.weights, hidden_states, encoder_hidden_states, temb, rotary_emb,
            values, candidate_cache_rows, update_cache, self.eps, debug, **options,
        )
        # 中间量仅供外部核验；普通调用返回与hidden同形状的输出。
        return result if debug else result["output"]
