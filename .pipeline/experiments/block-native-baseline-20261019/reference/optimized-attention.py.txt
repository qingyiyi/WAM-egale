"""延迟加载固定Torch2.9/SM80/BF16/D128的独立CUDA attention。"""
from functools import lru_cache
import importlib.util
from pathlib import Path
import torch

TORCH_COMMIT = "0fabc3ba44823f257e70ce397d989c8de5e362c1"


@lru_cache(maxsize=1)
def _extension():
    if torch.version.git_version != TORCH_COMMIT:
        raise RuntimeError("当前CUDA算子绑定固定Torch2.9版本")
    libraries = list(Path(__file__).parent.glob("wam_attention_cuda*.so"))
    if len(libraries) != 1:
        raise RuntimeError("请先在optimized目录执行 python build.py build_ext --inplace")
    spec = importlib.util.spec_from_file_location("wam_attention_cuda", libraries[0])
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def cuda_sdpa(q, k, v, scale=None):
    return _extension().sdpa(q, k, v, scale)
