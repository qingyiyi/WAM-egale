"""Build only the independent M37 device primitive; frozen DSOs are untouched."""
import os
from pathlib import Path
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

os.environ.setdefault('TORCH_CUDA_ARCH_LIST', '8.0')
root = Path(__file__).resolve().parent
headers = root.parents[2] / '.pipeline/cache/cutlass-v3.5.1/include'
setup(name='wam_contract_experimental', packages=[], ext_modules=[CUDAExtension(
    name='wam_contract_cuda', sources=[str(root/'csrc/contract_linear.cu')],
    include_dirs=[str(headers)],
    extra_compile_args={'cxx':['-O2'], 'nvcc':['-O3','-lineinfo']},
)], cmdclass={'build_ext':BuildExtension})
