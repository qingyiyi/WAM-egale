"""CPU-only builds of generated diagnostic copies, unique host/device identities."""
from pathlib import Path
import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

os.environ.setdefault('TORCH_CUDA_ARCH_LIST','8.0')
b=Path('/root/cxz/ops/whole-layer-warp-20261020/diagnostic')
setup(name='wam_whole_layer_finite_diagnostics',packages=[],ext_modules=[
    CUDAExtension(name='wam_whole_diagnostic_'+kind,sources=[str(b/(kind+'.cu'))],
        extra_compile_args={'cxx':['-O2'],'nvcc':['-O3','-lineinfo','-rdc=true']},
        dlink=True,dlink_libraries=['cudadevrt']) for kind in os.getenv('DIAG_KINDS','profile').split(',')],
    cmdclass={'build_ext':BuildExtension})
