"""单层Transformer的一次persistent launch。固定实验输入；请在inference_mode下调用。"""
from functools import lru_cache
import importlib.util
from pathlib import Path
import torch


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
        self.context_rows = self.context_roots = None
        self.context_kv = []
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

    def prepare_context(self, context, precompute=False):
        """在计时/Graph外准入固定context；内容变化后必须重新调用。

        分组只比较context的字节，不借用self-attention的历史行映射。
        这是实验静态接口，不是自动检测在线prompt变化的缓存协议。
        """
        self.context_rows = self.context_roots = None
        self.context_kv = []
        if context.shape[0] != 14:
            return
        bits = context.detach().contiguous().view(torch.uint8)
        roots, rows = [], []
        for row in range(14):
            group = next((g for g, root in enumerate(roots)
                          if torch.equal(bits[row], bits[root])), None)
            if group is None:
                group = len(roots)
                roots.append(row)
            rows.append(group)
        if len(roots) < 14:
            self.context_rows = torch.tensor(rows, dtype=torch.int64, device=context.device)
            self.context_roots = torch.tensor(roots, dtype=torch.int64, device=context.device)
        if precompute:
            compact = context if self.context_roots is None else context.index_select(0, self.context_roots)
            attention = self.block.attn2
            k0 = attention.to_k(compact)
            self.context_kv = [k0, attention.norm_k(k0), attention.to_v(compact)]

    def __call__(self, hidden_states, encoder_hidden_states, temb, rotary_emb,
                 update_cache=0, cache_name="pos", candidate_cache_rows=None,
                 debug=False):
        cache = self.block.attn1.attn_caches.get(cache_name)
        values = [] if cache is None else [
            cache[key] for key in ("k", "v", "id", "mask", "is_pred")
        ]
        extra = () if self.context_roots is None else (self.context_rows, self.context_roots)
        if self.context_kv:
            extra = (self.context_rows, self.context_roots, self.context_kv)
        result = extension().forward(
            self.weights, hidden_states, encoder_hidden_states, temb, rotary_emb,
            values, candidate_cache_rows, update_cache, self.eps, debug,
            *extra,
        )
        # 中间量仅供外部核验；普通调用返回与hidden同形状的输出。
        return result if debug else result["output"]
