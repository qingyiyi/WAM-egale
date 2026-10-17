"""New module namespace for one legacy ROOT configuration binding; numerical code unchanged."""
import ast,sys,types
from pathlib import Path


def install():
    if 'contract' in sys.modules:raise RuntimeError('legacy path contract unexpectedly imported before relocation binding')
    project=Path(__file__).resolve().parents[3]
    original=project/'投机采样/experiments/single_worker_chc_gpu_v4/contract.py'
    tree=ast.parse(original.read_bytes(),filename=str(original))
    assignments=[n for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(v,ast.Name) and v.id=='ROOT' for v in n.targets)]
    if len(assignments)!=1:raise RuntimeError('one source ROOT binding required')
    binding=assignments[0]
    if not (isinstance(binding.value,ast.Call) and isinstance(binding.value.func,ast.Name) and binding.value.func.id=='Path'
        and len(binding.value.args)==1 and isinstance(binding.value.args[0],ast.Constant)
        and binding.value.args[0].value=='/rjs/cxz/论文探索'):raise RuntimeError('frozen literal ROOT binding differs')
    binding.value.args[0]=ast.copy_location(ast.Constant(value=str(project)),binding.value.args[0])
    namespace=types.ModuleType('contract');namespace.__file__=str(original)
    namespace.__dict__['_baai_relocation_provenance']=dict(original_file=str(original),changed_binding='ROOT path only',old_value='/rjs/cxz/论文探索',new_value=str(project),mathematical_statements_unchanged=True,source_file_bytes_unchanged=True,no_SHA=True)
    sys.modules['contract']=namespace
    try:exec(compile(ast.fix_missing_locations(tree),str(original)+':baai-root-binding','exec'),namespace.__dict__)
    except BaseException:sys.modules.pop('contract',None);raise
