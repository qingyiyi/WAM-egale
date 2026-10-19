"""Build a selected CUDA component in place, without changing the environment."""
import os
from pathlib import Path
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

os.environ.setdefault('TORCH_CUDA_ARCH_LIST', '8.0')
ROOT = Path(__file__).resolve().parent
target = os.environ.get('WAM_BUILD_TARGET', 'attention')
source = {'attention': 'attention.cu', 'candidate': 'candidate_attention.cu',
          'ffn': 'ffn.cu'}[target]
prefix_headers = ROOT.parents[2] / '投机采样/experiments/shared_prefix_torch_v2/csrc'
gemm_headers = ROOT.parents[2] / '.pipeline/cache/cutlass-v3.5.1/include'
includes = ([str(gemm_headers)] if target == 'ffn' else []) + [
    str(ROOT / 'csrc/vendor/flash_attn'), str(ROOT / 'csrc/vendor/cutlass/include'), str(prefix_headers)]
setup(
    name='wam_operator_' + target, version='0.1.0', packages=[],
    ext_modules=[CUDAExtension(
        name='wam_' + target + '_cuda',
        sources=[str(ROOT / 'csrc' / source)],
        include_dirs=includes,
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
