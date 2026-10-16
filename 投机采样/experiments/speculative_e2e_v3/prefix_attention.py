"""Retain the frozen wrapper validation and explicit CPU oracle unchanged."""
import importlib.util
from pathlib import Path
import sys
path = Path(__file__).resolve().parent.parent / 'shared_prefix_gpu_v3/prefix_attention.py'
spec = importlib.util.spec_from_file_location('pinned_prefix_attention_contract_v3',path)
module = importlib.util.module_from_spec(spec); sys.modules[spec.name] = module; spec.loader.exec_module(module)
validate = module.validate
cpu_reference = module.cpu_reference
shared_attention = module.shared_attention
