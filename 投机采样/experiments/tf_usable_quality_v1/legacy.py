"""Explicit bridge to the verified legacy launch/cleanup and prefix modules."""
from __future__ import annotations
import importlib
from pathlib import Path
import sys

from tf_usable_quality_v1.protocol import EXP


def launcher():
    directory=EXP/'formal_e2e_speed_v1'
    # Legacy modules have fixed absolute imports. Verify each origin instead
    # of permitting a same-named module from another experiment to bind.
    for name, expected in (('speed_contract',directory/'speed_contract.py'),):
        if name in sys.modules and Path(sys.modules[name].__file__).resolve()!=expected:
            raise ImportError('legacy launch namespace collision: '+name)
    sys.path.insert(0,str(directory))
    module=importlib.import_module('formal_e2e_speed_v1.run')
    for name,expected in {
        'resource_guard':EXP/'single_worker_chc_gpu_v4/resource_guard.py',
        'ownership':EXP/'single_worker_chc_gpu_v4/ownership.py',
        'launch_identity':EXP/'speculative_e2e_v3/launch_identity.py',
        'cleanup':EXP/'morning_onpolicy_long8_v4/cleanup.py',
    }.items():
        if name not in sys.modules or Path(sys.modules[name].__file__).resolve()!=expected:
            raise ImportError('legacy ownership/cleanup namespace collision: '+name)
    return module.Campaign,module.cleanup_roots


def server_helpers():
    directory=EXP/'formal_e2e_speed_v1'
    if 'speed_contract' in sys.modules and Path(sys.modules['speed_contract'].__file__).resolve()!=directory/'speed_contract.py':
        raise ImportError('legacy server namespace collision')
    sys.path.insert(0,str(directory))
    module=importlib.import_module('formal_e2e_speed_v1.server_entry')
    module._paths()
    expected={'measured_online':EXP/'prefix_acceptance_v1/measured_online.py',
              'selection':EXP/'prefix_acceptance_v1/selection.py',
              'buffers':EXP/'prefix_acceptance_v1/buffers.py'}
    loaded={}
    for name,path in expected.items():
        found=importlib.util.find_spec(name)
        if found is None or Path(found.origin).resolve()!=path:
            raise ImportError('prefix module collision: '+name)
        loaded[name]=importlib.import_module(name)
    for name,path in {'adapter':EXP/'single_worker_chc_gpu_v4/adapter.py',
                      'contract':EXP/'single_worker_chc_gpu_v4/contract.py',
                      'paths':EXP/'prefix_acceptance_v1/paths.py'}.items():
        if name not in sys.modules or Path(sys.modules[name].__file__).resolve()!=path:
            raise ImportError('legacy prefix dependency namespace collision: '+name)
    return module.validate_server,module.RamAccounting,loaded
