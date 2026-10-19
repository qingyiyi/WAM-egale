"""Read only; no remote files, CUDA contexts or changes to running jobs."""
from pathlib import Path
import os,json,time,subprocess
os.chdir(Path.home()/'cxz');os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
BASE=Path('WAM-egale/投机采样/results/runtime')
def load(p):return json.loads(p.read_text()) if p.exists() else {}
def identity(d):
    try:
        p=Path('/proc')/str(d['pid']);f=(p/'stat').read_text().rsplit(')',1)[1].split()
        return dict(pid=d['pid'],birth_ticks=int(f[19]),uid=p.stat().st_uid,match=int(f[19])==d['birth_ticks'] and p.stat().st_uid==d['uid'])
    except (FileNotFoundError,KeyError):return dict(match=False)
def phase(p):
    status=load(p/'status.json');files=list(p.glob('ROW-*-RESULT.json'));rows=[load(x) for x in files]
    actual=[x for x in rows if x.get('complete_episode') and not x.get('no_new_episode')]
    alive=[]
    for q in p.glob('OWNED-*.json'):
        d=load(q);i=identity(d)
        if i['match']:alive.append(dict(**i,role=d.get('role')))
    out=dict(path=str(p),row_files=len(files),complete_rows=sum(bool(x.get('complete_episode')) for x in rows),actual_new_complete=len(actual),incomplete_rows=sum(not x.get('complete_episode',False) for x in rows),last_completed_age_seconds=time.time()-max((x.stat().st_mtime for x in files),default=time.time()),status=status.get('status'),heartbeat_age_seconds=time.time()-(p/'status.json').stat().st_mtime if (p/'status.json').exists() else None,current=status.get('current'),live_owned=alive,terminal=load(p/'SUPERVISOR-RESULT.json').get('status'))
    return out,rows
out=dict(at_UTC=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),suites={})
for suite in ['goal','spatial']:
    p=BASE/f'p0-tf-suite-{suite}-baai-chain-v1-20261018-r1';s=load(p/'status.json');mapping={};phases={};duplicates=[]
    for name in ['T_POOL','FINAL_0','FINAL_1']:
        info,rows=phase(p/'stages'/name);phases[name]=info
        for row in rows:
            if not row.get('complete_episode'):continue
            key=(row['case_id'],row['arm'])
            if key in mapping and any(mapping[key].get(k)!=row.get(k) for k in ['success','env_steps','failure800_penalty']):duplicates.append(key)
            mapping[key]=row
    common=sorted(set.intersection(*[{case for case,arm in mapping if arm==a} for a in ['T','C','P']]))
    out['suites'][suite]=dict(status=s.get('status'),controller=identity(load(p/'CONTROLLER-LAUNCHED.json')),active_steps=s.get('active_steps'),complete_common_case_groups=len(common),unique_complete_episodes=len(mapping),arm_complete={a:sum(arm==a for case,arm in mapping) for a in ['T','C','P']},common_success={a:sum(bool(mapping[case,a]['success']) for case in common) for a in ['T','C','P']},component_failures={a:sum(row.get('component_status','').startswith('FAILED') for (case,arm),row in mapping.items() if arm==a) for a in ['T','C','P']},duplicate_conflicts=duplicates,phases=phases)
p=BASE/'p0-object-capture-recovery-v3-20261019-r1';s=load(p/'status.json')
out['object']=dict(status=s.get('status'),controller=identity(load(p/'CONTROLLER-LAUNCH.json')),active=s.get('active'),completed=s.get('completed'),remaining_current_episodes=s.get('remaining_current_episodes'),heartbeat_age_seconds=time.time()-(p/'status.json').stat().st_mtime if (p/'status.json').exists() else None,terminal=load(p/'TERMINAL.json').get('status'))
out['object']['phases']={}
mapping={};conflicts=[]
for folder in [p/'CURRENT_DATA',p/'MATCHED_TARGET200',*sorted(p.glob('FINAL_*'))]:
    if not folder.is_dir():continue
    info,rows=phase(folder);out['object']['phases'][folder.name]=info
    if folder.name=='CURRENT_DATA':continue
    for row in rows:
        if not row.get('complete_episode'):continue
        key=(row['case_id'],row['arm'])
        if key in mapping and any(mapping[key].get(k)!=row.get(k) for k in ['success','env_steps','failure800_penalty']):conflicts.append(key)
        mapping[key]=row
common=sorted(set.intersection(*[{case for case,arm in mapping if arm==a} for a in ['T','C','P']]))
out['object'].update(complete_common_case_groups=len(common),unique_complete_final_episodes=len(mapping),
    arm_complete={a:sum(arm==a for case,arm in mapping) for a in ['T','C','P']},
    common_success={a:sum(bool(mapping[case,a]['success']) for case in common) for a in ['T','C','P']},
    duplicate_conflicts=conflicts)
r=subprocess.run(['nvidia-smi','--query-gpu=index,uuid,memory.used,memory.free,utilization.gpu','--format=csv,noheader'],capture_output=True,text=True,timeout=15);out['GPU']=r.stdout.strip()
print(json.dumps(out,ensure_ascii=False))
