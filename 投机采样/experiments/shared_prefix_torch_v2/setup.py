"""Local build only: no pip install and no GPU visibility during compilation."""
import os
from pathlib import Path
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
    raise RuntimeError('build must be CPU-only')
os.environ.setdefault('TORCH_CUDA_ARCH_LIST', '8.0')
ROOT = Path(__file__).resolve().parent
setup(name='speculative_shared_torch_v1', version='0.1.0',
      ext_modules=[CUDAExtension(
          name='_shared_torch_v1', sources=[str(ROOT / 'csrc/shared_torch.cu')],
          include_dirs=[str(ROOT / 'csrc/vendor/flash_attn'), str(ROOT / 'csrc/vendor/cutlass/include')],
          extra_compile_args={
              'cxx': ['-O2'],
              'nvcc': ['-O3', '-lineinfo', '-DUNFUSE_FMA',
                       '-DFLASHATTENTION_DISABLE_DROPOUT', '-DFLASHATTENTION_DISABLE_ALIBI',
                       '-DFLASHATTENTION_DISABLE_SOFTCAP', '-DFLASHATTENTION_DISABLE_LOCAL',
                       '-DFLASHATTENTION_DISABLE_UNEVEN_K'],
          })], cmdclass={'build_ext': BuildExtension})
import torch
if torch.cuda.is_initialized():
    raise RuntimeError('CPU build initialized CUDA')
print('CPU_BUILD_CUDA_INITIALIZED=false', flush=True)
