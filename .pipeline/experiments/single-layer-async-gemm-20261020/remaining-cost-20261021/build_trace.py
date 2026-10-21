from pathlib import Path
import sys
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension,CUDAExtension
B=Path(__file__).resolve().parent
R=Path('/root/cxz/ops/persistent-tile-20261019/workspace/算子研究/wam_ops/optimized')
sys.path.insert(0,str(R))
from fa2_codegen import generate
from norm_codegen import generate as norm
from cutlass_gemm_codegen import generate as gemm
gen=generate(B/'simple-stage-build/generated');norm(gen);cutlass=gemm(gen)
flags=['-O3','-lineinfo','-rdc=true','--ptxas-options=-v','-DWAM_CONTEXT_TILE_M=128','-DUNFUSE_FMA','-DFLASHATTENTION_DISABLE_DROPOUT','-DFLASHATTENTION_DISABLE_ALIBI','-DFLASHATTENTION_DISABLE_SOFTCAP','-DFLASHATTENTION_DISABLE_LOCAL','-DFLASHATTENTION_DISABLE_UNEVEN_K']
setup(name='wam_cost_trace_cuda',packages=[],ext_modules=[CUDAExtension(name='wam_cost_trace_cuda',sources=[str(B/'cost_trace.cu')],include_dirs=[str(gen),str(cutlass),str(R/'csrc'),str(R/'csrc/vendor/flash_attn'),str(R/'csrc/vendor/cutlass/include')],extra_compile_args={'cxx':['-O2'],'nvcc':flags},dlink=True,dlink_libraries=['cudadevrt'])],cmdclass={'build_ext':BuildExtension})
