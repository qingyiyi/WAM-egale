"""Build the complete native block; requires the existing own attention binary."""
import os
from pathlib import Path
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

os.environ.setdefault('TORCH_CUDA_ARCH_LIST', '8.0')
root = Path(__file__).resolve().parent
attention = list((root.parent / 'optimized').glob('wam_attention_cuda*.so'))
ffn = list((root.parent / 'optimized').glob('wam_ffn_cuda*.so'))
if len(attention) != 1 or len(ffn) != 1:
    raise RuntimeError('Build optimized attention and FFN CUDA components first')
setup(name='wam_native_block', packages=[], ext_modules=[CUDAExtension(
    name='wam_block_cuda', sources=[str(root / 'csrc/block.cu')],
    extra_objects=[str(attention[0]),str(ffn[0])], libraries=['cublasLt'],
    extra_compile_args={'cxx':['-O2'], 'nvcc':['-O3','-lineinfo']},
    extra_link_args=['-Wl,-rpath,'+str(attention[0].parent)],
)], cmdclass={'build_ext':BuildExtension})
