"""Generate observation-only copies from the pinned, unchanged sources."""
import ast
import hashlib
import json
from pathlib import Path

OWN=Path(__file__).resolve().parent
EXP=OWN.parent
ROOT=EXP.parent.parent
SNAPSHOT=ROOT/'review_exports/WAM-egale'
COMMIT='b0dfac3a02f65de33a1217246b83003615166640'

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    provenance={}
    for name in ('server_entry.py','client_entry.py','run.py'):
        src=EXP/'draft_freeze_quality_v1'/name
        pinned=SNAPSHOT/src.relative_to(ROOT)
        if src.read_bytes()!=pinned.read_bytes():raise ValueError('pinned source differs '+str(src))
        text=src.read_text().replace('draft_freeze_quality_v1.protocol','task4_direction_diagnostic_v1.protocol')
        text=text.replace('DRAFT_FREEZE_','DIRECTION_DIAG_')
        if name=='server_entry.py':
            text=text.replace('        proxy=online=parity=None',
                '        from task4_direction_diagnostic_v1.observer import Observer\n'
                '        diagnostic=Observer(s,job,arm)\n        proxy=online=parity=None')
            text=text.replace("            modules['buffers'].prepare(",
                "            diagnostic.install_action()\n            modules['buffers'].prepare(")
            text=text.replace('        original_reset,original_infer=s._reset,s.infer',
                '        diagnostic.install_io(accounting)\n        original_reset,original_infer=s._reset,s.infer')
        elif name=='client_entry.py':
            text=text.replace('    return client.main()',
                '    from task4_direction_diagnostic_v1.client_observer import install as install_diagnostic\n'
                '    install_diagnostic(client,out)\n    return client.main()')
        else:
            start=text.index('    def run(self):')
            stop=text.index('\ndef main():',start)
            text=text[:start]+'''    def run(self):
        failure=None
        try:
            for arm in self.plan['arms']:self.arm(arm)
            expected=[r['key'] for a in self.plan['arms'] for r in self.plan['cases'] if r['arm']==a]
            if self.completed!=expected:raise ValueError('four complete diagnostic episodes required')
            write(self.job/'COLLECTION.json',dict(status='COMPLETE_FOUR_DIAGNOSTIC_EPISODES',
                completed=self.completed,full_episodes=4,total_target_forwards=self.total_target_forwards,
                elapsed_sec=time.monotonic()-self.birth_monotonic,performance_claim=False))
        except BaseException as exc:
            failure=dict(status='FAILED_OR_PARTIAL_DIRECTION_DIAGNOSTIC',error=repr(exc),
                traceback=traceback.format_exc(),stage=self.stage,completed=self.completed,
                elapsed_sec=time.monotonic()-self.birth_monotonic)
        finally:
            try:
                closed=cleanup_roots(self.roots,self.pins,
                    deadline=min(time.monotonic()+30,self.final_deadline),wait_seconds=.5)
                write(self.job/'FINAL-OWNED-CLEANUP.json',closed)
                if closed.get('all_roots_reaped') is not True:raise RuntimeError('owned cleanup incomplete')
            except BaseException as exc:
                failure=failure or dict(status='FAILED_DIRECTION_DIAGNOSTIC_CLEANUP',error=repr(exc))
            self.nv.close()
        if failure is not None:
            write(self.job/'FAILED.json',failure)
            write(self.job/'status.json',failure,replace=True)
            raise RuntimeError(failure['error'])
        write(self.job/'status.json',dict(status='COMPLETE_COLLECTION_PENDING_CPU_DIAGNOSIS',
            completed=4,planned=4,performance_claim=False,permissions=self.plan['permissions']),replace=True)

'''+text[stop:]
            start=text.index('def _await_clock_guard(')
            stop=text.index('\n\nclass Campaign',start)
            text=text[:start]+'''def _await_clock_guard(job,plan,self_pin):
    from task4_direction_diagnostic_v1.clock import await_guard
    return await_guard(job,plan,self_pin)
'''+text[stop:]
            text=text.replace("    locks=[ROOT/", "    locks=[ROOT/")
        target=OWN/name
        target.write_text(text)
        provenance[name]=dict(source=str(src),source_sha256=sha(src),generated_sha256=sha(target))

    source=EXP.parent/'wam-p0-worktree/wan_va/wan_va_server.py'
    if source.read_bytes()!=(SNAPSHOT/source.relative_to(ROOT)).read_bytes():raise ValueError('original server changed')
    tree=ast.parse(source.read_text())
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='VA_Server')
    fn=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='_infer_action_verifier_k2_b2')
    gold=ast.dump(fn,include_attributes=False)
    def expr(call):return ast.parse('self._direction_diag.'+call).body[0]
    class Observe(ast.NodeTransformer):
        def visit_FunctionDef(self,node):
            node=self.generic_visit(node)
            if node.name=='record_update':
                index=next(i for i,x in enumerate(node.body) if isinstance(x,ast.If)
                    and ast.unparse(x.test)=='p0_trace_chunk is None')
                node.body.insert(index,expr('commit(step, state, direction, source, bonus_target_step_executed)'))
            return node
        def visit_Assign(self,node):
            if len(node.targets)!=1 or not isinstance(node.targets[0],ast.Name):return node
            name=node.targets[0].id
            calls={'remaining':'begin_round(step_index, remaining)',
                'projected':'projected(projected, timestep, offset)',
                'tree':'tree(tree, step_index)',
                'scores':'score(tree, scores, step_index)',
                'execution':'executed(tree, scores, route, execution, step_index)'}
            return [node,expr(calls[name])] if name in calls else node
        def visit_Continue(self,node):return [expr('end_round(step_index)'),node]
        def visit_While(self,node):
            node=self.generic_visit(node);node.body.append(expr('end_round(step_index)'));return node
    fn=Observe().visit(fn)
    fn.body.insert(1,expr('begin_action(actions, action_timesteps, frame_st_id, action_update_count)'))
    fn.body.insert(len(fn.body)-1,expr('end_action(actions)'))
    class Remove(ast.NodeTransformer):
        def visit_Expr(self,node):
            if (isinstance(node.value,ast.Call) and isinstance(node.value.func,ast.Attribute)
                    and isinstance(node.value.func.value,ast.Attribute)
                    and node.value.func.value.attr=='_direction_diag'):return None
            return node
    import copy
    restored=Remove().visit(copy.deepcopy(fn))
    if ast.dump(restored,include_attributes=False)!=gold:raise ValueError('observation AST reverse proof failed')
    ast.fix_missing_locations(fn)
    (OWN/'action_observed.py').write_text(ast.unparse(fn)+'\n')
    provenance['action_observed.py']=dict(source=str(source),source_sha256=sha(source),
        original_AST_sha256=hashlib.sha256(gold.encode()).hexdigest(),reverse_AST_exact=True,
        generated_sha256=sha(OWN/'action_observed.py'))
    (OWN/'GENERATION.json').write_text(json.dumps(dict(commit=COMMIT,files=provenance),indent=2)+'\n')
    print(json.dumps(dict(status='GENERATED_FROM_PINNED_SOURCES_REVERSE_AST_EXACT',files=len(provenance))))

if __name__=='__main__':main()
