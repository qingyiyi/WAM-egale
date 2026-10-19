"""构建独立子图DSO；旧build.py及baseline依赖的DSO保持冻结。"""
import os
from pathlib import Path

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

os.environ.setdefault('TORCH_CUDA_ARCH_LIST', '8.0')
root = Path(__file__).resolve().parent
cutlass = root.parents[2] / '.pipeline/cache/cutlass-v3.5.1/include'
setup(
    name='wam_isolated_subgraphs', packages=[],
    ext_modules=[CUDAExtension(
        name='wam_optimized_block_cuda',
        sources=[str(root / 'csrc' / name) for name in
                 ('block.cu', 'attention_tiles.cu', 'ffn_tiles.cu')],
        include_dirs=[str(root / 'csrc/vendor/flash_attn'), str(cutlass)],
        libraries=['cublasLt'],
        # Preserve the respective frozen block/attention arithmetic flags.
        extra_compile_args={'cxx': ['-O2'], 'nvcc': ['-O3', '-lineinfo']},
    )],
    cmdclass={'build_ext': BuildExtension},
)
