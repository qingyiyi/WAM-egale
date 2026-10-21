from pathlib import Path
import sys
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
B = Path(__file__).resolve().parent
R = Path('/root/cxz/ops/persistent-tile-20261019/workspace/算子研究/wam_ops/optimized')
label = sys.argv.pop(1)
assert label in ('before', 'after')
sys.path.insert(0, str(R))
from fa2_codegen import generate
from norm_codegen import generate as norm
from cutlass_gemm_codegen import generate as gemm
generated = generate(B / ('build-' + label) / 'generated')
norm(generated)
cutlass = gemm(generated)
name = 'wam_gemm_' + label + '_profile_cuda'
flags = ['-O3', '-lineinfo', '-rdc=true', '--ptxas-options=-v', '-DWAM_CONTEXT_TILE_M=128',
         '-DUNFUSE_FMA', '-DFLASHATTENTION_DISABLE_DROPOUT', '-DFLASHATTENTION_DISABLE_ALIBI',
         '-DFLASHATTENTION_DISABLE_SOFTCAP', '-DFLASHATTENTION_DISABLE_LOCAL',
         '-DFLASHATTENTION_DISABLE_UNEVEN_K']
setup(name=name, packages=[], ext_modules=[CUDAExtension(name=name,
    sources=[str(B / ('profile_' + label + '.cu'))],
    include_dirs=[str(generated), str(cutlass), str(R / 'csrc'),
                  str(R / 'csrc/vendor/flash_attn'), str(R / 'csrc/vendor/cutlass/include')],
    extra_compile_args={'cxx': ['-O2'], 'nvcc': flags}, dlink=True, dlink_libraries=['cudadevrt'])],
    cmdclass={'build_ext': BuildExtension})
