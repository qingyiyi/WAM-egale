"""Original production driver with narrow reversible pending-only bookkeeping."""
import argparse
import ast
import copy
import fcntl
import inspect
import os
from pathlib import Path
import signal
import socket
import sys
import textwrap

OWN=Path(__file__).resolve().parent;EXP=OWN.parent
if str(EXP) not in sys.path:sys.path.insert(0,str(EXP))
from tf_effect_campaign_continue_v2.protocol import load,read,write,require,sha,RT
from tf_effect_campaign_v1 import run as original


def build_execute():
    tree=ast.parse(textwrap.dedent(inspect.getsource(original.execute)));gold=copy.deepcopy(tree)
    changes={};counts={};added=[]
    def key(node):return ast.dump(node,include_attributes=False)
    def replacement(old,new,name):
        changes[key(new)]=copy.deepcopy(old);counts[name]=counts.get(name,0)+1;return new
    class Change(ast.NodeTransformer):
        def visit_For(self,node):
            node=self.generic_visit(node)
            if isinstance(node.target,ast.Name) and node.target.id=='arm' and ast.unparse(node.iter)=="p['arms']":
                node.iter=replacement(node.iter,ast.parse("[a for a in p['arms'] if any(r['arm']==a and r['key'] in p['pending_keys'] for r in p['cases'])]",mode='eval').body,'arms')
            elif isinstance(node.target,ast.Name) and node.target.id=='row' and isinstance(node.iter,ast.ListComp):
                node.iter=replacement(node.iter,ast.parse("[r for r in p['cases'] if r['arm']==arm and r['key'] in p['pending_keys']]",mode='eval').body,'rows')
            return node
        def visit_Assign(self,node):
            node=self.generic_visit(node)
            names=[x.id for x in node.targets if isinstance(x,ast.Name)]
            if names==['expected']:
                node.value=replacement(node.value,ast.parse("[r['key'] for a in p['arms'] for r in p['cases'] if r['arm']==a and r['key'] in p['pending_keys']]",mode='eval').body,'expected')
            elif names==['ref']:
                node.value=replacement(node.value,ast.parse("job/'T-READY.json' if (job/'T-READY.json').exists() else Path(p['target_initial_rng_reference'])",mode='eval').body,'reference')
            return node
        def visit_Call(self,node):
            node=self.generic_visit(node)
            if isinstance(node.func,ast.Name) and node.func.id=='len' and len(node.args)==1 and ast.unparse(node.args[0])=="p['cases']":
                return replacement(node,ast.parse("len(p['pending_keys'])",mode='eval').body,'planned')
            if isinstance(node.func,ast.Name) and node.func.id=='write' and len(node.args)>=2:
                name=ast.unparse(node.args[0]);payload=node.args[1]
                if name in ("job / 'COLLECTION.json'","job / 'SUPERVISOR-RESULT.json'"):
                    require(isinstance(payload,ast.Call) and ast.unparse(payload.func)=='dict','original terminal payload changed')
                    old=copy.deepcopy(payload)
                    status=next(k for k in payload.keywords if k.arg=='status')
                    if name=="job / 'COLLECTION.json'":status.value=ast.Constant(value='COMPLETE_PENDING_EPISODES_ONLY')
                    else:
                        require(isinstance(status.value,ast.IfExp),'original supervisor status conditional changed')
                        status.value.orelse=ast.Constant(value='COMPLETE_PENDING_EPISODES_ONLY')
                    for field in ('predecessor_job','imported_terminal_manifest','pending_keys','cumulative_budget'):
                        payload.keywords.append(ast.keyword(arg=field,value=ast.parse("p['"+field+"']",mode='eval').body))
                    payload.keywords.append(ast.keyword(arg='continuation_only',value=ast.Constant(value=True)))
                    replacement(old,payload,'terminal')
            return node
    tree=Change().visit(tree)
    # All arms, including T, must match this DEV's actual target-only RNG.
    for node in ast.walk(tree):
        if isinstance(node,ast.If) and ast.unparse(node.test)=="(job / f'{arm}-READY.json').exists()":
            require_line=ast.parse("require(ready['target_initial_rng']==read(Path(p['target_initial_rng_reference']))['target_initial_rng'],'continuation target initialization differs from original DEV T READY')").body[0]
            index=next(i for i,n in enumerate(node.body) if isinstance(n,ast.Assign))+1
            # Preserve original READY assertions and append only a read-only check.
            node.body.insert(index,require_line);added.append(key(require_line))
    require(counts==dict(arms=1,rows=1,expected=1,reference=1,planned=1,terminal=2)
        and len(added)==1,'narrow original runner adaptation sites changed')
    class Restore(ast.NodeTransformer):
        def visit(self,node):
            if node is None:return None
            dumped=key(node)
            if dumped in added:return None
            if dumped in changes:return self.visit(copy.deepcopy(changes[dumped]))
            return super().visit(node)
    restored=Restore().visit(copy.deepcopy(tree))
    require(key(restored)==key(gold),'inverse AST proof failed: production calls/math changed')
    ast.fix_missing_locations(tree);namespace=dict(vars(original))
    # OWN intentionally remains the ORIGINAL directory: server/client sources,
    # complete CAL setup, transport and production inference are not replaced.
    exec(compile(tree,str(__file__)+':pending-only-driver','exec'),namespace)
    return namespace['execute'],dict(status='PASS_REVERSIBLE_PENDING_ONLY_RUNNER_AST',
        original_source=str(Path(original.__file__).resolve()),original_sha256=sha(original.__file__),
        changes=counts,extra_target_RNG_assertion=1,original_server_client_and_model_operations_unchanged=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--job',type=Path,required=True);args=parser.parse_args()
    job=args.job.resolve();p=load(job)
    require(not (job/'RUN-STARTED.json').exists(),'one attempt, never overwrite original or continuation failure')
    execute,proof=build_execute();write(job/'RUNNER-AST-PROOF.json',proof)
    locks=[]
    try:
        for port in (p['port'],p['master_port']):
            with socket.socket() as sock:sock.bind(('127.0.0.1',port))
        for name in ('.speculative_minimal_gpu.lock','.shared-prefix-GPU.lock','.target-trace-draft-dev-v1.lock'):
            lock=(RT/name).open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(lock)
        signal.signal(signal.SIGTERM,lambda *_: (_ for _ in ()).throw(SystemExit('owned stop')))
        execute(job,p)
    except BaseException as exc:
        if not (job/'FAILED.json').exists():write(job/'FAILED.json',dict(status='FAILED_PENDING_ONLY_CONTINUATION',error=repr(exc),continuation_only=True))
        if not (job/'SPAWN-ATTEMPTED.json').exists() and not (job/'FINAL-OWNED-CLEANUP.json').exists():
            write(job/'FINAL-OWNED-CLEANUP.json',dict(all_roots_reaped=True,no_child_launch_attempted=True))
        if not (job/'SUPERVISOR-RESULT.json').exists():
            no_child=not (job/'SPAWN-ATTEMPTED.json').exists()
            write(job/'SUPERVISOR-RESULT.json',dict(status='FAILED_OR_PARTIAL',failure=dict(error=repr(exc)),
                completed=[] if no_child else None,full_episodes=0 if no_child else None,
                total_target_forwards=0 if no_child else 'UNKNOWN_SEE_CLOCK_AND_CASE_COUNTERS',
                cleanup=read(job/'FINAL-OWNED-CLEANUP.json') if (job/'FINAL-OWNED-CLEANUP.json').exists() else 'UNKNOWN_SEE_CLOCK',
                observed_pins=[],continuation_only=True,predecessor_job=p['predecessor_job'],
                imported_terminal_manifest=p['imported_terminal_manifest'],pending_keys=p['pending_keys'],
                cumulative_budget=p['cumulative_budget'],permissions=p['permissions'],performance_claim=False))
        raise
    finally:
        for lock in reversed(locks):fcntl.flock(lock,fcntl.LOCK_UN);lock.close()


if __name__=='__main__':main()
