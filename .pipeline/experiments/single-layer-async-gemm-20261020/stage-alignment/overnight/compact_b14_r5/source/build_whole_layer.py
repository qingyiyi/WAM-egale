"""Build current optimized whole block with internal pinned FA2 device tiles."""
import os
from pathlib import Path
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
from fa2_codegen import generate
from norm_codegen import generate as generate_norm
from cutlass_gemm_codegen import generate as generate_gemm
os.environ.setdefault('TORCH_CUDA_ARCH_LIST','8.0')
root=Path(__file__).resolve().parent
build=Path(os.environ['WAM_FA2_BUILD_ROOT'])
generated=generate(build/'generated')
generate_norm(generated)
gemm_headers=generate_gemm(generated)
# The bounded real-fixture comparison selected 128 context rows. Action
# projections remain 64 rows; explicit 64 reproduces the registered control.
context_rows=int(os.environ.get('WAM_CONTEXT_TILE_M','128'))
extensions=[]
for compact in ((True,) if os.environ.get("WAM_COMPACT_ONLY") == "1" else (False, True)):
    name = 'wam_whole_b14_cuda' if compact else 'wam_whole_fa2_cuda'
    flags = ['-O3','-lineinfo','-rdc=true','--ptxas-options=-v',
             f'-DWAM_CONTEXT_TILE_M={context_rows}', f'-DWAM_COMPACT_B14={int(compact)}',
             '-DUNFUSE_FMA','-DFLASHATTENTION_DISABLE_DROPOUT',
             '-DFLASHATTENTION_DISABLE_ALIBI','-DFLASHATTENTION_DISABLE_SOFTCAP',
             '-DFLASHATTENTION_DISABLE_LOCAL','-DFLASHATTENTION_DISABLE_UNEVEN_K']
    if compact: flags.append('-Dwam_whole_layer_v1=wam_whole_layer_compact14')
    extensions.append(CUDAExtension(name=name,sources=[str(root/'csrc/whole_layer.cu')],
        include_dirs=[str(generated),str(gemm_headers),str(root/'csrc'),
                      str(root/'csrc/vendor/flash_attn'),str(root/'csrc/vendor/cutlass/include')],
        extra_compile_args={'cxx':['-O2'],'nvcc':flags},dlink=True,dlink_libraries=['cudadevrt']))
setup(name='wam_whole_fa2',packages=[],ext_modules=extensions,
      cmdclass={'build_ext':BuildExtension})
