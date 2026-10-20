"""Create a RoboTwin-only adapter from exact source functions; canonical source stays read-only."""
from pathlib import Path
import ast,copy,json,shutil
B=Path(__file__).resolve().parent
LOCAL=Path('/rjs/cxz/论文探索')
P=LOCAL/'投机采样'
source=P/'wam-p0-worktree/wan_va/wan_va_server.py'
tree=ast.parse(source.read_text());cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='VA_Server')
names={'_prepare_action_candidate_input','_repeat_action_candidates_for_cfg','_p0_action_channel_ids','_p0_action_compact',
       '_p0_action_scatter_compact','_p0_action_scatter_compact_batch','_p0_action_project_state',
       '_infer_action_verifier_k2_b2','_target_action_candidate_directions'}
fns=[copy.deepcopy(n) for n in cls.body if isinstance(n,ast.FunctionDef) and n.name in names]
assert {n.name for n in fns}==names
# These functions remain original math. Runtime wrappers provide only observation,
# fixed predictor, exact production selection and original O2/O3 workspace helpers.
imports='import torch\nimport numpy as np\nfrom einops import rearrange\nfrom mesh_helper import get_mesh_id\nfrom bonus_helper import execute_verified_k2_bonus_step\n'
(B/'server_math.py').write_text(imports+'\n\n'.join(ast.unparse(n) for n in fns)+'\n')
# Keep exact selector functions without importing a legacy experiment launcher.
for src,dst,names in [(P/'experiments/bonus_acceptance_v2/selection.py','depth_selector.py',{'depth','choose'}),
                      (P/'experiments/prefix_acceptance_v1/selection.py','prefix_selector.py',{'depth','choose'})]:
 t=ast.parse(src.read_text());nodes=[n for n in t.body if isinstance(n,ast.FunctionDef) and n.name in names]
 head='import math\nNAME="prefix_then_cost"\nimport depth_selector as baseline\n' if dst.startswith('prefix') else 'import math\nNAME="depth_then_cost"\n'
 (B/dst).write_text(head+'\n\n'.join(ast.unparse(n) for n in nodes)+'\n')
for src,dst in [(P/'scripts/p0_batched_verify.py','tree_api.py'),(P/'wam-p0-worktree/wan_va/utils/p0_k3_b2.py','bonus_helper.py'),
                (P/'experiments/prefix_acceptance_v1/scalar.py','scalar.py'),(P/'experiments/prefix_acceptance_v1/buffers.py','buffers.py')]:shutil.copy2(src,B/dst)
(B/'template.py').write_text('''import types

def clone_function(fn, **overrides):
    result=types.FunctionType(fn.__code__, {**fn.__globals__, **overrides}, fn.__name__, fn.__defaults__, fn.__closure__)
    result.__kwdefaults__=fn.__kwdefaults__;result.__annotations__=fn.__annotations__;result.__doc__=fn.__doc__;result.__qualname__=fn.__qualname__
    return result
''')
# Kernel geometry is a separately named Q32 entry, not removal of old Q16 guard.
c=B/'q32';c.mkdir(exist_ok=True)
old=P/'experiments/shared_prefix_torch_v2/csrc'
s=(old/'shared_torch.cu').read_text().replace('speculative_flash','robotwin_flash_q32')
s=s.replace('{14, 16, 24, 128}','{14, 32, 24, 128}').replace('[14,16,24,128]','[14,32,24,128]')
s=s.replace('p.seqlen_q = 16;','p.seqlen_q = 32;').replace('{14,24,16}','{14,24,32}').replace('cv, 14, 16','cv, 14, 32').replace('slots.numel() + 16','slots.numel() + 32')
(c/'shared_q32.cu').write_text(s)
for p in old.glob('*.h'):shutil.copy2(p,c/p.name)
print('generated selected server AST and Q32 kernel only')
