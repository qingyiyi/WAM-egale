"""真实fixture的静态强基线；不是动态生产cache协议或新调度器。

admit阶段固定有效slot/条件和原GEMM形状。Graph replay不会重新执行Python
guard，任何条件或cache布局变化都必须重新admit。候选的私有KV从不共享。
"""
from contextlib import contextmanager
from functools import lru_cache
import importlib.util
from pathlib import Path
import types

import torch
from .attention import TORCH_COMMIT

ROOT = Path(__file__).resolve().parents[3]


@lru_cache(None)
def extension(name):
    if torch.version.git_version != TORCH_COMMIT:
        raise RuntimeError('扩展绑定固定Torch2.9源码版本')
    directory = (ROOT / '投机采样/experiments/shared_prefix_torch_v2'
                 if name == '_shared_torch_v1' else Path(__file__).parent)
    libraries = list(directory.glob(name + '*.so'))
    if len(libraries) != 1:
        raise RuntimeError(f'{name}: 需要唯一的已构建扩展')
    spec = importlib.util.spec_from_file_location(name, libraries[0])
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def guarded_group(prefix):
    # 保留旧基线规则：不使原来的K64尾块增加一个tile。
    return 2 if (prefix + 32 + 63) // 64 == (prefix + 16 + 63) // 64 else 1


class PreparedBlock:
    def __init__(self, block, arguments, kwargs, cache):
        if kwargs['update_cache'] != 0:
            raise ValueError('此入口只支持已登记只读fixture')
        self.block, self.arguments, self.kwargs, self.cache = block, arguments, kwargs, cache
        self.slots = cache['mask'].nonzero(as_tuple=False).squeeze(-1)
        self.rows = kwargs.get('candidate_cache_rows')
        self.context = arguments[1]
        self.context_copy = self.context.clone()
        self.cached = None
        self.original_self = block.attn1.forward
        self.original_k, self.original_v = block.attn2.to_k.forward, block.attn2.to_v.forward
        if self.rows is None:
            free = (~cache['mask']).nonzero(as_tuple=False).squeeze(-1)
            self.private_slots = free[:arguments[0].shape[1]]
            if self.private_slots.numel() != arguments[0].shape[1]:
                raise ValueError('静态入口不支持cache驱逐')
            mask = cache['mask'].clone()
            mask[self.private_slots] = True
            self.all_slots = mask.nonzero(as_tuple=False).squeeze(-1)
            self.next_id = cache['id'][self.slots].max() + 1
            self.private_ids = self.next_id.expand(self.private_slots.numel()).contiguous()
        elif not torch.equal(self.rows, torch.tensor([0] * 7 + [1] * 7, device=self.rows.device)):
            raise ValueError('当前真实B14 fixture必须保留CFG分组')

    def fill(self):
        # 用原形状原Linear冷填充；不从B2复制成B14、不改变GEMM形状。
        self.cached = (self.original_k(self.context).detach().clone(),
                       self.original_v(self.context).detach().clone())
        return sum(value.numel() * value.element_size() for value in self.cached)

    def admit(self):
        if not torch.equal(self.context, self.context_copy):
            raise ValueError('静态条件发生变化，必须重新填充')
        valid = self.cache['mask'].nonzero(as_tuple=False).squeeze(-1)
        if not torch.equal(valid, self.slots):
            raise ValueError('有效历史布局发生变化，必须重新admit')

    def forward(self, attention, q, k, v, rotary_emb, update_cache=0,
                cache_name='pos', candidate_cache_rows=None):
        # 与原WanAttention保留相同投影、RMSNorm、RoPE和BF16边界。
        query, key, value = attention.to_q(q), attention.to_k(k), attention.to_v(v)
        query = attention.norm_q(query).unflatten(2, (attention.heads, -1))
        key = attention.norm_k(key).unflatten(2, (attention.heads, -1))
        value = value.unflatten(2, (attention.heads, -1))
        if rotary_emb is not None:
            def rotate(x):
                z = torch.view_as_complex(x.to(torch.float64).reshape(*x.shape[:3], -1, 2))
                return torch.view_as_real(z * rotary_emb).flatten(3).to(x.dtype)
            query, key = rotate(query), rotate(key)
        if self.rows is None:
            c, slots = self.cache, self.private_slots
            c['k'][:, slots], c['v'][:, slots] = key, value
            c['id'].index_copy_(0, slots, self.private_ids)
            c['is_pred'].index_fill_(0, slots, False)
            # 原update0在返回前把private mask还原为false；不改变有效历史。
            output = attention.attn_op(query, c['k'][:, self.all_slots], c['v'][:, self.all_slots])
        elif self.engine == 'native':
            output = extension('_shared_torch_v1').shared(query, self.cache['k'],
                self.cache['v'], key, value, self.slots, self.rows)
        elif self.engine == 'guarded':
            output = extension('wam_candidate_cuda').grouped(query, self.cache['k'],
                self.cache['v'], key, value, self.slots, guarded_group(self.slots.numel()))
        else:
            hk = self.cache['k'][:, self.slots].index_select(0, self.rows)
            hv = self.cache['v'][:, self.slots].index_select(0, self.rows)
            output = attention.attn_op(query, torch.cat((hk, key), 1), torch.cat((hv, value), 1))
        output = output.flatten(2, 3).type_as(query)
        return attention.to_out[1](attention.to_out[0](output))

    @contextmanager
    def activate(self, engine='dense', cached=False):
        self.admit()
        if engine not in ('dense', 'native', 'guarded') or (self.rows is None and engine != 'dense'):
            raise ValueError('该形状不支持此强基线')
        if cached and self.cached is None:
            raise ValueError('必须先用原Linear冷填充')
        self.engine = engine
        self.block.attn1.forward = types.MethodType(self.forward, self.block.attn1)
        if cached:
            self.block.attn2.to_k.forward = lambda x: self.cached[0]
            self.block.attn2.to_v.forward = lambda x: self.cached[1]
        try:
            yield
        finally:
            # 删除实例覆盖，恢复原类方法；不改全局类或源码。
            del self.block.attn1.forward
            if cached:
                del self.block.attn2.to_k.forward
                del self.block.attn2.to_v.forward
