from pathlib import Path
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension,CUDAExtension

here=Path(__file__).resolve().parent
setup(name='wam_m38_static',packages=[],ext_modules=[CUDAExtension(
    name='wam_whole_m38_static',sources=[str(here/'static.cu')],
    extra_compile_args={'cxx':['-O2'],'nvcc':['-O3','-lineinfo','-rdc=true']},
    dlink=True,dlink_libraries=['cudadevrt'])],cmdclass={'build_ext':BuildExtension})
