"""Same shared-prefix math, separate SM120 architecture/device admission."""
from pathlib import Path
import difflib,json,os,shutil,subprocess,sys
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension,CUDAExtension
P=Path('/root/autodl-tmp/cxz/spec-speed-20261025')
kind='robotwin' if '--robotwin' in sys.argv else 'libero'
V=P/'bundle/project/投机采样/experiments/shared_prefix_torch_v2/csrc/vendor'
S=P/'bundle/robotwin/runtime/q32' if kind=='robotwin' else V.parent
B=P/'port'/kind;B.mkdir(parents=True,exist_ok=True)
assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and os.environ['TORCH_CUDA_ARCH_LIST']=='12.0'
original=(S/('shared_q32.cu' if kind=='robotwin' else 'shared_torch.cu')).read_text()
old='props->major == 8 && props->minor == 0 && props->multiProcessorCount == 108'
assert original.count(old)==1
new=original.replace(old,'props->major == 12 && props->minor == 0').replace('pinned A100 SM80/108SM contract required','independent RTX5090 SM120 migration contract required')
(B/'shared_torch.cu').write_text(new)
for header in S.glob('*.h'):
    shutil.copyfile(header,B/header.name)
if not (B/'vendor').exists():(B/'vendor').symlink_to(V,target_is_directory=True)
(P/'results'/('SHARED-SM120-RESOURCE-DIFF-'+kind+'.patch')).write_text(''.join(difflib.unified_diff(original.splitlines(True),new.splitlines(True),fromfile='copied-A100-device-admission',tofile='temporary-SM120-device-admission')))
os.chdir(B)
setup(name='wam-temporary-sm120-shared',version='0.0.1',script_args=['build_ext','--inplace'],ext_modules=[CUDAExtension(name='_robotwin_shared_q32' if kind=='robotwin' else '_shared_torch_v1',sources=[str(B/'shared_torch.cu')],include_dirs=[str(S),str(V/'flash_attn'),str(V/'cutlass/include')],extra_compile_args={'cxx':['-O2'],'nvcc':['-O3','-lineinfo','-DUNFUSE_FMA','-DFLASHATTENTION_DISABLE_DROPOUT','-DFLASHATTENTION_DISABLE_ALIBI','-DFLASHATTENTION_DISABLE_SOFTCAP','-DFLASHATTENTION_DISABLE_LOCAL','-DFLASHATTENTION_DISABLE_UNEVEN_K']})],cmdclass={'build_ext':BuildExtension})
