from pathlib import Path
import os,sys
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension,CUDAExtension
R=Path('/root/cxz/ops/persistent-tile-20261019/workspace/算子研究/wam_ops/optimized');B=Path(__file__).resolve().parent
sys.path.insert(0,str(R))
from fa2_codegen import generate
from norm_codegen import generate as generate_norm
gen=generate(B/'generated');generate_norm(gen)
setup(name='whole_stage_profile',packages=[],ext_modules=[CUDAExtension(name='wam_stage_profile_cuda',sources=[str(B/'profile.cu')],include_dirs=[str(B),str(gen),str(R/'csrc'),str(R/'csrc/vendor/flash_attn'),str(R/'csrc/vendor/cutlass/include')],extra_compile_args={'cxx':['-O2'],'nvcc':['-O3','-lineinfo','-rdc=true','-DUNFUSE_FMA','-DFLASHATTENTION_DISABLE_DROPOUT','-DFLASHATTENTION_DISABLE_ALIBI','-DFLASHATTENTION_DISABLE_SOFTCAP','-DFLASHATTENTION_DISABLE_LOCAL','-DFLASHATTENTION_DISABLE_UNEVEN_K']},dlink=True,dlink_libraries=['cudadevrt'])],cmdclass={'build_ext':BuildExtension})
