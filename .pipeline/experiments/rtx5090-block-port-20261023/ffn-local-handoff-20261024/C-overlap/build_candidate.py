"""Build current optimized whole block with internal pinned FA2 device tiles."""
import os
from pathlib import Path
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
from fa2_codegen import generate
from norm_codegen import generate as generate_norm
from cutlass_gemm_codegen import generate as generate_gemm
os.environ.setdefault('TORCH_CUDA_ARCH_LIST','12.0')
root=Path('/root/autodl-tmp/cxz/operator-sm120-20261023/workspace/算子研究/wam_ops/optimized')
build=Path(os.environ['WAM_FA2_BUILD_ROOT'])
if os.environ.get('WAM_REUSE_GENERATED') == '1':
    generated=build/'generated'
    gemm_headers=root.parents[2]/'.pipeline/cache/cutlass-v3.5.1/include'
else:
    generated=generate(build/'generated')
    generate_norm(generated)
    gemm_headers=generate_gemm(generated)
# The bounded real-fixture comparison selected 128 context rows. Action
# projections remain 64 rows; explicit 64 reproduces the registered control.
context_rows=int(os.environ.get('WAM_CONTEXT_TILE_M','128'))
setup(name='wam_whole_ffn_overlap',packages=[],ext_modules=[CUDAExtension(
 name='wam_whole_ffn_overlap_cuda',sources=[str(Path(__file__).resolve().parent/'whole_layer.cu')],
 include_dirs=[str(generated),str(gemm_headers),str(root/'csrc'),str(root/'csrc/vendor/flash_attn'),str(root/'csrc/vendor/cutlass/include')],
 extra_compile_args={'cxx':['-O2'],'nvcc':['-O3','-lineinfo','-rdc=true','--ptxas-options=-v',f'-DWAM_CONTEXT_TILE_M={context_rows}',
 '-DUNFUSE_FMA','-DFLASHATTENTION_DISABLE_DROPOUT','-DFLASHATTENTION_DISABLE_ALIBI',
 '-DFLASHATTENTION_DISABLE_SOFTCAP','-DFLASHATTENTION_DISABLE_LOCAL','-DFLASHATTENTION_DISABLE_UNEVEN_K']},
 libraries=['cuda'],dlink=True,dlink_libraries=['cudadevrt'])],cmdclass={'build_ext':BuildExtension})
