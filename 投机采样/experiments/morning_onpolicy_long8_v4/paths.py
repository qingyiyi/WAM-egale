"""Load frozen private helpers without editing their files or globals."""
import importlib.util
from pathlib import Path
import sys

OWN = Path(__file__).resolve().parent
ROOT = OWN.parents[2]
EXP = OWN.parent
BASE = EXP / 'single_worker_chc_gpu_v4'
S0 = EXP / 'speculative_e2e_v3'
WORKTREE = ROOT / '投机采样/wam-p0-worktree'


def load(name, path):
    path = Path(path).resolve()
    if name in sys.modules:
        module = sys.modules[name]
        if Path(module.__file__).resolve() != path:
            raise ImportError('private helper identity changed: ' + name)
        return module
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


for directory in (S0, BASE):
    if str(directory) not in sys.path:
        sys.path.append(str(directory))
