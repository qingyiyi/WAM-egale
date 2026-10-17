"""Run only never-attempted Object rows; keep the failed row and phase incomplete."""
from pathlib import Path
import argparse,ast,copy,sys,traceback

OWN=Path(__file__).resolve().parent
sys.dont_write_bytecode=True
sys.path.insert(0,str(OWN.parent))
from tf_suite_pipeline_v1 import runner as original
from tf_suite_pipeline_v1.protocol import load,read,write,require,PERMISSIONS


def unstarted_execute():
    path=Path(original.__file__);tree=ast.parse(path.read_bytes(),filename=str(path))
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Runner')
    method=copy.deepcopy(next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='execute'))
    loops=[n for n in ast.walk(method) if isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id=='spec']
    require(len(loops)==1,'one source controller row loop required')
    require(ast.unparse(loops[0].iter)=="self.p['rows']",'original row loop changed')
    loops[0].iter=ast.parse("[r for r in self.p['rows'] if r['row_id'] not in self.p['resume_blocked_row_ids']]",mode='eval').body
    ns=dict(vars(original));module=ast.fix_missing_locations(ast.Module(body=[method],type_ignores=[]))
    exec(compile(module,str(path)+':unstarted-row-controller/v1','exec'),ns)
    return ns['execute']


class UnstartedRunner(original.Runner):
    execute=unstarted_execute()

    def guard(self,stage):
        super().guard(stage)
        f=self.job/'server-worker0/LIVE-COUNTER.json'
        attempts=read(f)['target_forward_attempts'] if f.exists() else 0
        require(attempts+self.p['prior_target_forward_debit']<=801000,'old plus new T_POOL attempts exceeded original budget')


def main():
    a=argparse.ArgumentParser();a.add_argument('--job',type=Path,required=True);x=a.parse_args();job=x.job.resolve(strict=True)
    try:
        p=load(job);require(p['phase']=='T_POOL' and p['suite']=='libero_object' and p['resume_blocked_row_ids']==[0], 'only predeclared unstarted Object rows allowed')
        old=Path(p['original_partial_source_job']);require(read(old/'FINAL-OWNED-CLEANUP.json')['all_owned_groups_empty'],'original failed phase must be closed')
        actual=read(old/'server-worker0/LIVE-COUNTER.json')
        require(actual['active_target_forwards']==0 and actual['target_forward_attempts']==p['prior_target_forward_debit']==296,'original actual attempt debit differs')
        require(not any((old/'episodes').glob('case-*/*/EPISODE-TERMINAL.json')),'unexpected prior outcome; stop for identity review')
        require(p['no_training_until_complete_original_FIT_CAL'] is True,'collection is not a relaxed training trigger')
        return UnstartedRunner(job,p).execute()
    except BaseException as e:
        if not (job/'RUNNER-ENTRY-FAILED.json').exists():write(job/'RUNNER-ENTRY-FAILED.json',dict(error=repr(e),traceback=traceback.format_exc(),permissions=PERMISSIONS))
        raise

if __name__=='__main__':raise SystemExit(main())
