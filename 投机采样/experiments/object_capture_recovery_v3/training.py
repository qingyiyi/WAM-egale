"""Explicit path-only migration adapter; every original data assertion is kept."""
from pathlib import Path
import ast
import copy
import json
import sys

OWN=Path(__file__).resolve().parent
sys.dont_write_bytecode=True
sys.path.insert(0,str(OWN.parent))
from tf_suite_pipeline_v1 import training as original


def migrated_pack(collection_job,job,registration):
    import torch
    plan=original.read(Path(collection_job)/'plan.json')
    mappings=plan['migration_context_out_prefix_mapping']
    translations=[]
    def load_with_host_context(path,**kwargs):
        payload=torch.load(path,**kwargs)
        original.require(payload['schema']=='tf-suite-draft-chunk-columns/v1','only original suite shard path binding')
        old=payload['context']['out']
        for mapping in mappings:
            prefix=mapping['original']
            if old==prefix or old.startswith(prefix+'/'):
                actual=mapping['actual']+old[len(prefix):]
                payload=dict(payload,context=dict(payload['context'],out=actual))
                translations.append(dict(shard=str(path),original_out=old,actual_out=actual,
                    tensor_objects_changed=False,only_context_out_string_rebound=True))
                break
        return payload
    path=OWN.parent/'tf_suite_pipeline_v1/training.py'
    fn=next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='pack_collection')
    class BindOneLoad(ast.NodeTransformer):
        count=0
        def visit_Call(self,node):
            node=self.generic_visit(node)
            if isinstance(node.func,ast.Attribute) and isinstance(node.func.value,ast.Name) and node.func.value.id=='torch' and node.func.attr=='load':
                self.count+=1;node.func=ast.Name(id='__migration_load',ctx=ast.Load())
            return node
    change=BindOneLoad();bound=change.visit(copy.deepcopy(fn))
    original.require(change.count==1,'exactly one original shard load, no guard removal')
    namespace=dict(original.__dict__,__migration_load=load_with_host_context)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[bound],type_ignores=[])),str(path)+':host-context-only','exec'),namespace)
    value=namespace['pack_collection'](collection_job,job,registration)
    original.write(Path(job)/'MIGRATION-CONTEXT-TRANSLATIONS.json',dict(
        schema='object-data-host-context-binding/v1',mappings=mappings,translations=translations,
        original_assertions_kept=True,old_shard_files_not_modified=True,no_tensor_field_rewritten=True))
    return value


if __name__=='__main__':
    original.pack_collection=migrated_pack
    result=original.main()
    print(json.dumps({k:v for k,v in result.items() if k not in ('registration_content','loss_log')},ensure_ascii=False),flush=True)
