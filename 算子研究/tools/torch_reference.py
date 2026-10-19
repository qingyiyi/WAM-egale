"""只读Torch attention oracle；不属于native forward。"""
from torch.nn import functional as F


def torch_sdpa(q, k, v):
    return F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2),
                                         v.transpose(1, 2)).transpose(1, 2)
