"""CPU-only local extension build, pinned existing SM80 vendor sources."""
import os
from pathlib import Path
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension,CUDAExtension
assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
os.environ['TORCH_CUDA_ARCH_LIST']='8.0'
OWN=Path(__file__).resolve().parent
VENDOR=OWN.parent/'shared_prefix_torch_v2/csrc'
setup(name='wam_candidate_tile_research',version='0.1.0',
 ext_modules=[CUDAExtension(name='_wam_candidate_tile',sources=[str(OWN/'csrc/group.cu')],
 include_dirs=[str(VENDOR),str(VENDOR/'vendor/flash_attn'),str(VENDOR/'vendor/cutlass/include')],
 extra_compile_args={'cxx':['-O2'],'nvcc':['-O3','-lineinfo','-DUNFUSE_FMA',
 '-DFLASHATTENTION_DISABLE_DROPOUT','-DFLASHATTENTION_DISABLE_ALIBI',
 '-DFLASHATTENTION_DISABLE_SOFTCAP','-DFLASHATTENTION_DISABLE_LOCAL','-DFLASHATTENTION_DISABLE_UNEVEN_K']})],
 cmdclass={'build_ext':BuildExtension})
import torch
assert not torch.cuda.is_initialized()
