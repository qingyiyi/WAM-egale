"""New single-worker Long timing protocol; original robot/model math is reused."""
import argparse
import ast
from datetime import datetime
import inspect
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
import time

APP=Path('/root/cxz/WAM-egale/投机采样/experiments')
BASE=Path('/root/cxz/performance/libero-four-suite-20261019/isolated')
sys.dont_write_bytecode=True
sys.path.insert(0,str(APP))
from tf_fullsuite_eval_v2 import protocol


def load_timing(job):
    job=Path(job).resolve(strict=True);p=protocol.read(job/'plan.json')
    reg=protocol.read(BASE/'REGISTRATION.json');old=protocol.read(Path(p['original_source_plan']))
    assert job.is_relative_to(BASE) and p['schema']=='wam-single-worker-long-timing/v1'
    assert p['frozen_permissions']==protocol.PERMISSIONS==reg['permissions']
    assert p['timing_case_ids']==list(range(490,500)) and p['active_worker']==1
    assert p['selected']==old['selected'] and p['normalizer']==old['normalizer']
    for k in ['bank','target_checkpoint_path','target_checkpoint_id','registry','mode',
              'selector','threshold','evaluation_seed','horizon','target_source_reference']:
        assert p[k]==old[k],k
    assert p['horizon']==800 and p['threshold']==0.007839680526450805
    assert p['absolute_work_stop']==reg['absolute_work_stop']
    assert p['absolute_cleanup_deadline']==reg['absolute_cleanup_deadline']
    assert p['worker_forward_caps']=={'0':0,'1':121000,'2':0,'3':0}
    assert p['registered_new_phase_attempt_cap']==121000
    assert len(p['cases'])==500 and p['prior_completed_case_ids']==[]
    for cid in p['timing_case_ids']:
        row=p['cases'][cid]
        assert (row['case_id'],row['task'],row['init'])==(cid,cid%10,49)
        assert row['execution_worker_index']==1 and row['proposed_GPU_index']==1
        assert row['within_case_order']==('T_C_P','C_P_T','P_T_C')[cid%3]
        assert row['execution_authorized'] is True
    assert Path(p['job']).resolve()==job and p['per_episode_wall_sec']==1200
    assert p['per_episode_forward_attempt_cap']==4000
    for key in ['bank','registry','target_source_reference','va_python','client_python']:
        assert Path(p[key]).exists(),key
    assert Path(p['selected']['path']).exists() and Path(p['target_checkpoint_path']).exists()
    return p


protocol.load=load_timing


def run_worker(role, args):
    if role=='server':
        from baai_resume_v1.path_binding import install
        install()
    script=APP/'tf_fullsuite_eval_v2'/f'{role}_entry.py'
    sys.argv=[str(script),*args]
    runpy.run_path(str(script),run_name='__main__')


def runner(job):
    # The legacy constructor checks that four assignments are representable.
    # Only worker1 is started below; actual server/client affinity stays two each.
    os.sched_setaffinity(0,set(range(32)))
    from tf_fullsuite_eval_v2 import runner as original
    from baai_suite_pipeline_v1.runtime import environment

    class SingleWorker(original.Runner):
        def __init__(self, job, p):
            super().__init__(job,p)
            self.affinity[1]=dict(server=[12,13],client=[14,15])
            os.sched_setaffinity(0,{21})

        def spawn(self,argv,log,env,affinity,role):
            if role.startswith(('server-','client-')):
                kind='server' if role.startswith('server-') else 'client'
                argv=[argv[0],'-B',Path(__file__),kind,*argv[3:]]
                env=environment(self.p,env,interpreter=argv[0])
            return super().spawn(argv,log,env,affinity,role)

        def waves(self):
            for cid in self.p['timing_case_ids']:
                self.guard('new_registered_isolated_case')
                if (self.job/'STOP_AFTER_EPISODE').exists():break
                if time.time()+3960+900>=self.hard:break
                if self.run_case(cid)['status']!='COMPLETE_RECORDED_CASE':break

        def run_report(self):
            self.report_result=dict(returncode=0,kind='NEW_REGISTERED_TEN_CASE_TIMING_SUBSET',
                complete_cases=[c for c,v in self.results.items() if v['status']=='COMPLETE_RECORDED_CASE'],
                not_independent_quality_samples=True,permissions=protocol.PERMISSIONS)
            protocol.write(self.job/'TIMING-SUBSET-REPORT.json',self.report_result)

    # Reuse all real readiness checks. Only the two worker enumeration nodes
    # change from four workers to the one registered worker; no assertion drops.
    tree=ast.parse(inspect.getsource(original.Runner.start_servers).lstrip())
    class OneWorker(ast.NodeTransformer):
        replaced=0
        def visit_Call(self,node):
            if isinstance(node.func,ast.Name) and node.func.id in ('range','set'):
                if node.func.id=='range' and len(node.args)==1 and isinstance(node.args[0],ast.Constant) and node.args[0].value==4:
                    self.replaced+=1
                    return ast.copy_location(ast.Tuple(elts=[ast.Constant(value=1)],ctx=ast.Load()),node)
            return self.generic_visit(node)
    change=OneWorker();tree=change.visit(tree);ast.fix_missing_locations(tree)
    assert change.replaced==2,'original start_servers enumeration changed'
    ns=dict(original.Runner.start_servers.__globals__)
    exec(compile(tree,str(APP/'tf_fullsuite_eval_v2/runner.py')+':single-worker-resource-binding','exec'),ns)
    SingleWorker.start_servers=ns['start_servers']
    p=load_timing(job);work=SingleWorker(job,p)
    code=work.execute()
    old=protocol.read(job/'SUPERVISOR-RESULT.json')
    complete=sorted(c for c,v in work.results.items() if v['status']=='COMPLETE_RECORDED_CASE')
    passed=(complete==p['timing_case_ids'] and old['all_owned_groups_empty'] and old['failure'] is None)
    # The original supervisor still reports its 500-case expectation as PARTIAL.
    # Preserve it; this new schema evaluates only the newly registered 10 cases.
    protocol.write(job/'BENCHMARK-PHASE-RESULT.json',dict(
        schema='new-isolated-long-timing-subset/v1',
        status='COMPLETE_REGISTERED_ISOLATED_PANEL' if passed else 'PARTIAL_ISOLATED_PANEL',
        complete_rows=3*len(complete),complete_cases=complete,legacy_returncode=code,
        legacy_500_supervisor_status=old['status'],legacy_supervisor_unchanged=True,
        cleanup=old['cleanup'],permissions=protocol.PERMISSIONS,no_quality_certificate=True))
    return 0 if passed else 2


def launch(job):
    p=load_timing(job);assert not (job/'RUN-STARTED.json').exists()
    assert time.time()<datetime.fromisoformat(p['absolute_work_stop']).timestamp()
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',
             OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    children=[]; processes=[]
    for role,argv in [('guard',[p['va_python'],'-B',str(APP/'tf_fullsuite_eval_v2/guard.py'),'--job',str(job)]),
                      ('runner',[p['va_python'],'-B',str(Path(__file__)),'runner','--job',str(job)])]:
        with (job/f'{role}.log').open('x') as f:
            child=subprocess.Popen(argv,cwd=job,env=env,stdout=f,stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,start_new_session=True)
        stat=Path(f'/proc/{child.pid}/stat').read_text().rsplit(')',1)[1].split()
        pin=dict(pid=child.pid,birth_ticks=int(stat[19]),pgid=child.pid,uid=os.getuid(),
                 owner_job=str(job),role=role,argv=argv)
        protocol.write(job/('GUARD-PROCESS.json' if role=='guard' else f'OWNED-{child.pid}.json'),pin)
        children.append(pin)
        processes.append(child)
        if role=='guard':
            until=time.time()+15
            while not (job/'GUARD-STARTED.json').exists():
                assert child.poll() is None and time.time()<until
                time.sleep(.1)
    protocol.write(job/'LAUNCH-RESULT.json',dict(status='OWNED_TIMING_RUNNER_AND_GUARD_STARTED',children=children))
    print(json.dumps(children))
    # Keep the parent alive to reap the runner before the independent guard
    # checks its lifetime.  A zombie is not an active worker, but the legacy
    # guard deliberately does not guess this from a missing context.
    hard=datetime.fromisoformat(p['absolute_cleanup_deadline']).timestamp()
    exits=[]
    for child in reversed(processes):
        child.wait(timeout=max(1,hard-time.time()+20));exits.append(dict(pid=child.pid,exit_code=child.returncode))
    protocol.write(job/'LAUNCH-PARENT-RESULT.json',dict(owned_children_reaped=True,exits=exits))
    return 0 if all(x['exit_code']==0 for x in exits) else 2


def main():
    role=sys.argv.pop(1)
    if role in ('server','client'):return run_worker(role,sys.argv[1:])
    p=argparse.ArgumentParser();p.add_argument('--job',type=Path,required=True)
    job=p.parse_args().job.resolve(strict=True)
    return launch(job) if role=='launch' else runner(job) if role=='runner' else 2


if __name__=='__main__':raise SystemExit(main())
