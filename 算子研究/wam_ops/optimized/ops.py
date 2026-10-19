"""独立优化版原语入口；不加载baseline或旧共享动态库。"""
from .native import extension


def cuda_sdpa(q, k, v, scale=None):
    return extension().sdpa(q, k, v, scale, False)
