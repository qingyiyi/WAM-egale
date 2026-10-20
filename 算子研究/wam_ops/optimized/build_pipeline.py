"""Build unique research DSOs; never rebuild baseline/v1 shared dependencies."""
import os
from pathlib import Path
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension,CUDAExtension
from pipeline_codegen import generate

os.environ.setdefault('TORCH_CUDA_ARCH_LIST','8.0')
root=Path(os.environ['WAM_PIPELINE_BUILD_ROOT'])
kinds=os.environ.get('WAM_PIPELINE_KINDS','sync,async,profile').split(',')
generated=generate(root/'generated',kinds,root/'profile_instrument.py')
setup(name='wam_whole_pipeline',packages=[],ext_modules=[CUDAExtension(
    name='wam_whole_pipeline_'+kind,sources=[str(generated/(kind+'.cu'))],
    include_dirs=[str(Path(__file__).parent/'csrc')],
    extra_compile_args={'cxx':['-O2'],'nvcc':['-O3','-lineinfo','-rdc=true','--ptxas-options=-v']},
    dlink=True,dlink_libraries=['cudadevrt']) for kind in kinds],cmdclass={'build_ext':BuildExtension})
