"""Retain frozen admission, zero-copy history, mutation and lease guards."""
import importlib.util
from pathlib import Path
import sys
path = Path(__file__).resolve().parent.parent / 'shared_prefix_gpu_v3/prefix_adapter.py'
spec = importlib.util.spec_from_file_location('pinned_prefix_adapter_contract_v3',path)
module = importlib.util.module_from_spec(spec); sys.modules[spec.name] = module; spec.loader.exec_module(module)
SharedPrefix = module.SharedPrefix
LeaseError = module.LeaseError
