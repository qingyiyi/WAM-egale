"""The shared CUDA path reads CFG2 plus each candidate's current KV directly."""
import os
from pathlib import Path
import sys


def run(q,pk,pv,ck,cv,slots,rows):
    from task_contract import ENABLED,APPROVED
    if os.environ.get(ENABLED) != '1' or os.environ.get(APPROVED) != '1':
        raise ValueError('native model experiment is default off')
    native = Path(__file__).resolve().parent.parent / 'shared_prefix_torch_v2'
    if str(native) not in sys.path: sys.path.append(str(native))
    import _shared_torch_v1
    return _shared_torch_v1.shared(q,pk,pv,ck,cv,slots,rows)
