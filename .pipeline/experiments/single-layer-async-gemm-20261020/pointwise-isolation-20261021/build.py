from pathlib import Path
import sys
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension,CUDAExtension
B=Path(__file__).resolve().parent
R=Path('/root/cxz/ops/persistent-tile-20261019/workspace/算子研究/wam_ops/optimized')
sys.path.insert(0,str(R))
from cutlass_gemm_codegen import generate
headers=generate(B/'generated')
setup(name='wam_isolate',packages=[],ext_modules=[CUDAExtension(name='wam_isolate_cuda',sources=[str(B/'isolate.cu')],include_dirs=[str(B),str(B/'generated'),str(headers)],extra_compile_args={'cxx':['-O2'],'nvcc':['-O3','-lineinfo','--ptxas-options=-v','-DUNFUSE_FMA','-DWAM_CONTEXT_TILE_M=128']})],cmdclass={'build_ext':BuildExtension})
