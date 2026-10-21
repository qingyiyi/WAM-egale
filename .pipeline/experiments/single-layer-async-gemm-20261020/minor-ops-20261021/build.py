from pathlib import Path
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension,CUDAExtension
B=Path(__file__).resolve().parent
O=Path('/root/cxz/ops/persistent-tile-20261019/workspace/算子研究/wam_ops/optimized')
s=(O/'csrc/norm.cuh').read_text();a=s.index('template <typename T, typename T_ACC, bool rms_norm = false,');z=s.index('template <typename T, typename T_ACC',a+1)
s=s[a:z].replace('vectorized_layer_norm_kernel_impl(', 'minor_rms_forward(').replace('T* Y){','T* Y, float* s_data){').replace('extern __shared__ float s_data[];', '// supplied scratch').replace('auto i1 = blockIdx.x;', 'auto i1 = 0;').replace('rstd[i1] = rstd_val;','if(rstd)rstd[i1] = rstd_val;')
(B/'minor_norm.cuh').write_text('// Derived pinned Torch; license in norm.cuh.\n#pragma once\nnamespace wam_optimized_norm {\n'+s+'\n}\n')
setup(name='wam_minor',packages=[],ext_modules=[CUDAExtension(name='wam_minor_cuda',sources=[str(B/'minor.cu')],include_dirs=[str(B),str(O/'csrc')],extra_compile_args={'cxx':['-O2'],'nvcc':['-O3','-lineinfo','--ptxas-options=-v','-DUNFUSE_FMA']})],cmdclass={'build_ext':BuildExtension})
