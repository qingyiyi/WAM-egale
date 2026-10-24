from pathlib import Path
import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension,CUDAExtension
assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
B=Path(__file__).resolve().parent
V=Path('/root/cxz/WAM-egale/投机采样/experiments/shared_prefix_torch_v2/csrc/vendor')
(B/'q32/vendor').symlink_to(V,target_is_directory=True) if not (B/'q32/vendor').exists() else None
setup(name='robotwin_shared_q32',packages=[],ext_modules=[CUDAExtension(name='_robotwin_shared_q32',
 sources=[str(B/'q32/shared_q32.cu')],include_dirs=[str(V/'flash_attn'),str(V/'cutlass/include')],
 extra_compile_args={'cxx':['-O2'],'nvcc':['-O3','-lineinfo','-DUNFUSE_FMA','-DFLASHATTENTION_DISABLE_DROPOUT',
 '-DFLASHATTENTION_DISABLE_ALIBI','-DFLASHATTENTION_DISABLE_SOFTCAP','-DFLASHATTENTION_DISABLE_LOCAL',
 '-DFLASHATTENTION_DISABLE_UNEVEN_K']})],cmdclass={'build_ext':BuildExtension})
