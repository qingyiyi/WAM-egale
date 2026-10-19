# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
"""原参数与cache兼容容器；完整数学链位于baseline/csrc/block.cu。"""
import torch


class Attention(torch.nn.Module):

    def __init__(self, dim, heads=8, dim_head=64, eps=1e-05, dropout=0.0, cross_attention_dim_head=None, attn_mode='torch'):
        super().__init__()
        if attn_mode != 'torch':
            raise ValueError('native仅支持已对齐的Torch参数协议')
        from ..optimized.attention import cuda_sdpa
        self.attn_op = cuda_sdpa
        self.inner_dim = dim_head * heads
        self.heads = heads
        self.cross_attention_dim_head = cross_attention_dim_head
        self.kv_inner_dim = self.inner_dim if cross_attention_dim_head is None else cross_attention_dim_head * heads
        self.to_q = torch.nn.Linear(dim, self.inner_dim, bias=True)
        self.to_k = torch.nn.Linear(dim, self.kv_inner_dim, bias=True)
        self.to_v = torch.nn.Linear(dim, self.kv_inner_dim, bias=True)
        self.to_out = torch.nn.ModuleList([torch.nn.Linear(self.inner_dim, dim, bias=True), torch.nn.Dropout(dropout)])
        self.norm_q = torch.nn.RMSNorm(dim_head * heads, eps=eps, elementwise_affine=True)
        self.norm_k = torch.nn.RMSNorm(dim_head * heads, eps=eps, elementwise_affine=True)
        self.attn_caches = {} if cross_attention_dim_head is None else None
        self.cache_mode = 'original'

    def clear_pred_cache(self, cache_name):
        if self.attn_caches is not None:
            cache = self.attn_caches[cache_name]
            cache['mask'][cache['is_pred']] = False

    def clear_cache(self, cache_name):
        if self.attn_caches is not None:
            self.attn_caches[cache_name] = None

    def init_kv_cache(self, cache_name, total_tolen, num_head, head_dim, device, dtype, batch_size):
        if self.attn_caches is not None:
            self.attn_caches[cache_name] = self._original_cache(total_tolen, num_head, head_dim, device, dtype, batch_size)

    @staticmethod
    def _original_cache(total_tolen, num_head, head_dim, device, dtype, batch_size):
        return {'k': torch.empty([batch_size, total_tolen, num_head, head_dim], device=device, dtype=dtype), 'v': torch.empty([batch_size, total_tolen, num_head, head_dim], device=device, dtype=dtype), 'id': torch.full((total_tolen,), -1, device=device), 'mask': torch.zeros((total_tolen,), dtype=torch.bool, device=device), 'is_pred': torch.zeros((total_tolen,), dtype=torch.bool, device=device)}

    def forward(self, *args, **kwargs):
        raise RuntimeError("Attention仅为参数/cache容器；请调用完整native TransformerBlock")
