"""构建同计算体调度诊断；产物只写入指定实验目录。"""
from pathlib import Path
import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

root = Path(__file__).resolve().parent
build = Path(os.environ['WAM_RUN_DIR'])
headers = root.parents[2] / '.pipeline/cache/cutlass-v3.5.1/include'
source = (headers / 'cutlass/gemm/threadblock/mma_multistage.h').read_text()
assert source.count('__syncthreads();') == 2
source = source.replace('MmaMultistage', 'WamMmaMultistage').replace(
    '__syncthreads();', 'asm volatile("bar.sync 2, %0;" :: "n"('
    '32 * Base::WarpCount::kM * Base::WarpCount::kN * Base::WarpCount::kK) : "memory");')
gen = build / 'generated'
gen.mkdir(exist_ok=True)
(gen / 'cutlass_gemm_mainloop.cuh').write_text(source)
setup(name='wam_projection_schedule_cuda', packages=[], ext_modules=[CUDAExtension(
    name='wam_projection_schedule_cuda', sources=[str(root/'csrc/projection_schedule.cu')],
    include_dirs=[str(gen), str(headers), str(root/'csrc')],
    extra_compile_args={'cxx':['-O2'], 'nvcc':['-O3','-lineinfo','--ptxas-options=-v',
                                            '-DWAM_CONTEXT_TILE_M=256']})],
    cmdclass={'build_ext': BuildExtension})
