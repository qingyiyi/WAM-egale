"""Build the single CUDA operator in place, without pip/system installation."""
import os
from pathlib import Path
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

os.environ.setdefault('TORCH_CUDA_ARCH_LIST', '8.0')
ROOT = Path(__file__).resolve().parent
setup(
    name='wam_operator_attention', version='0.1.0', packages=[],
    ext_modules=[CUDAExtension(
        name='wam_attention_cuda',
        sources=[str(ROOT / 'csrc/attention.cu')],
        include_dirs=[str(ROOT / 'csrc/vendor/flash_attn'), str(ROOT / 'csrc/vendor/cutlass/include')],
        extra_compile_args={
            'cxx': ['-O2'],
            'nvcc': ['-O3', '-lineinfo', '-DUNFUSE_FMA',
                     '-DFLASHATTENTION_DISABLE_DROPOUT', '-DFLASHATTENTION_DISABLE_ALIBI',
                     '-DFLASHATTENTION_DISABLE_SOFTCAP',
                     '-DFLASHATTENTION_DISABLE_LOCAL', '-DFLASHATTENTION_DISABLE_UNEVEN_K'],
        },
    )],
    cmdclass={'build_ext': BuildExtension},
)
