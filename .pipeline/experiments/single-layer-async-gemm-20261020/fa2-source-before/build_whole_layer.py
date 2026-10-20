"""独立 SM80 整层实验 DSO；不重编 baseline/M32 的任何依赖。"""
import os
from pathlib import Path

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

os.environ.setdefault('TORCH_CUDA_ARCH_LIST', '8.0')
root = Path(__file__).resolve().parent
setup(name='wam_whole_layer_experimental', packages=[], ext_modules=[CUDAExtension(
    name='wam_whole_layer_cuda', sources=[str(root / 'csrc/whole_layer.cu')],
    extra_compile_args={'cxx': ['-O2'], 'nvcc': ['-O3', '-lineinfo', '-rdc=true']},
    dlink=True, dlink_libraries=['cudadevrt'],
)], cmdclass={'build_ext': BuildExtension})
