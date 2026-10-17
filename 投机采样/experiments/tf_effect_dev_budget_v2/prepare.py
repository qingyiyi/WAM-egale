"""Prospective DEV budget forecast correction; quality/cases/math unchanged.

Collection's frozen producer is untouched. The one allowed modification to the
existing preparation function is DEV cap5h ->7h, still bounded by master12h.
The original cap proposal remains in the master record and amendments are
bound before DEV. The user's later09:00 report deadline only tightens caps.
"""
import argparse,ast,copy,inspect,sys,textwrap
from pathlib import Path
OWN=Path(__file__).resolve().parent;EXP=OWN.parent;sys.path.insert(0,str(EXP))
from tf_effect_campaign_v1 import protocol as base

def main():
    a=argparse.ArgumentParser();a.add_argument('--job',type=Path,required=True);a.add_argument('--audit',type=Path,required=True)
    a.add_argument('--training',type=Path,required=True);a.add_argument('--trigger',type=Path,required=True);args=a.parse_args()
    amendment=base.RT/'p0-tf-effect-closure-v1-20261016-r1/DEV-BUDGET-FORECAST-AMENDMENT.json'
    decision=base.read(amendment)
    base.require(decision['new_DEV_wall_cap_sec']==25200 and decision['quality_rules_unchanged'] and decision['registered_before_new_training_and_DEV'],'prospective finite amendment required')
    user_amendment=base.RT/'p0-tf-effect-closure-v1-20261016-r1/USER-0900-DEADLINE-AMENDMENT.json'
    user_limit=base.read(user_amendment)
    base.require(user_limit['registered_before_DEV'] and user_limit['quality_cases_seed_training_unchanged'] and user_limit['earlier_existing_caps_still_bind'],'prospective earlier user deadline required')
    tree=ast.parse(textwrap.dedent(inspect.getsource(base.prepare)));gold=copy.deepcopy(tree)
    matches=[n for n in ast.walk(tree) if isinstance(n,ast.Constant) and n.value==18000]
    base.require(len(matches)==1,'unique DEV5h preparation constant required');matches[0].value=25200
    fn=tree.body[0]
    idx=next(i for i,n in enumerate(fn.body) if isinstance(n,ast.Expr) and isinstance(n.value,ast.Call)
        and isinstance(n.value.func,ast.Attribute) and isinstance(n.value.func.value,ast.Name)
        and n.value.func.value.id=='job' and n.value.func.attr=='mkdir')
    notes=ast.parse("""
limit=read(USER_AMENDMENT)
user_gpu_hard=datetime.fromisoformat(limit['DEV_GPU_cleanup_by'])
user_analysis_hard=datetime.fromisoformat(limit['analysis_cleanup_by'])
effective_hard=min(datetime.fromisoformat(p['absolute_cleanup_deadline']),user_gpu_hard)
effective_global=min(datetime.fromisoformat(p['global_cleanup_deadline']),user_analysis_hard)
require((effective_hard-now).total_seconds()>900,'earlier user deadline leaves insufficient DEV admission budget')
p.update(budget_amendment=str(AMENDMENT),user_deadline_amendment=str(USER_AMENDMENT),
    budget_profile='DEV7h_within_master12h_and_user0900',
    absolute_cleanup_deadline=effective_hard.isoformat(),
    absolute_work_stop=(effective_hard-timedelta(seconds=p['cleanup_reserve_sec'])).isoformat(),
    global_cleanup_deadline=effective_global.isoformat(),user_report_deadline=limit['report_requested_by'])
p['evidence_sha256'][str(AMENDMENT)]=sha(AMENDMENT)
p['evidence_sha256'][str(USER_AMENDMENT)]=sha(USER_AMENDMENT)
p['source_sha256'][str(PREPARE_SOURCE)]=sha(PREPARE_SOURCE)
p['source_sha256'][str(LAUNCH_SOURCE)]=sha(LAUNCH_SOURCE)
p['source_sha256'][str(REPORT_SOURCE)]=sha(REPORT_SOURCE)
""").body
    fn.body[idx:idx]=notes
    restored=copy.deepcopy(tree);del restored.body[0].body[idx:idx+len(notes)]
    replacements=[n for n in ast.walk(restored) if isinstance(n,ast.Constant) and n.value==25200]
    base.require(len(replacements)==1,'one finite cap modification');replacements[0].value=18000
    base.require(ast.dump(restored,include_attributes=False)==ast.dump(gold,include_attributes=False),'case/quality preparation AST must remain unchanged')
    ast.fix_missing_locations(tree);namespace={**vars(base),'AMENDMENT':amendment,'USER_AMENDMENT':user_amendment,'PREPARE_SOURCE':Path(__file__).resolve(),
        'LAUNCH_SOURCE':OWN.parent/'tf_effect_job_launch_v1/launch.py',
        'REPORT_SOURCE':OWN.parent/'tf_effect_quality_report_v1/report.py'}
    exec(compile(tree,str(__file__)+':budget-only-preparation','exec'),namespace)
    p=namespace['prepare'](args.job,collect=False,audit=args.audit,training=args.training,trigger=args.trigger)
    print(dict(status='REGISTERED_FINITE_DEV7H',job=p['job'],episodes=len(p['cases']),absolute_cleanup_deadline=p['absolute_cleanup_deadline']))
if __name__=='__main__':main()
