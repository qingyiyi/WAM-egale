"""Bounded CPU-only closure of the already authorized final NCU attempt.

No GPU jobs, model edits, retries, process termination, or quality promotion.
Publish only if the public mirror and remote main still match the registered HEAD.
"""
from pathlib import Path
import datetime
import hashlib
import json
import os
import subprocess
import sys
import time
import traceback

B=Path(__file__).resolve().parent
R=B.parents[3]
N=B/'local-ncu-final-r2'
M=R/'review_exports/WAM-egale-libero-fullscale-20261017'
os.sched_setaffinity(0,{4,5})
job=json.loads((B/'CLOSURE-JOB.json').read_text())
out=dict(status='WAITING_FOR_NCU_TERMINAL',start=time.time(),GPU_jobs_started=0)
def save():
    p=B/'CLOSURE-STATUS.json';q=p.with_suffix('.tmp');q.write_text(json.dumps(out,indent=2)+'\n');q.replace(p)
def run(command,**kwargs):
    p=subprocess.run(command,cwd=R,text=True,capture_output=True,timeout=120,**kwargs)
    if p.returncode:
        raise RuntimeError(f'{command[0:2]} rc={p.returncode}: {p.stderr[-1500:]}')
    return p.stdout
try:
    assert not (B/'CLOSURE-STATUS.json').exists()
    out['owned_guard']=dict(pid=os.getpid(),birth_ticks=int(Path(f'/proc/{os.getpid()}/stat').read_text().rsplit(')',1)[1].split()[19]))
    save()
    end=datetime.datetime.fromisoformat(job['deadline']).timestamp()
    while time.time()<end:
        try:s=json.loads((N/'STATUS.json').read_text())
        except json.JSONDecodeError:
            time.sleep(1);continue
        if s['status'] in ('COMPLETE','FAILED_OR_PARTIAL','EXPIRED_WITHOUT_GPU_WINDOW'):
            break
        time.sleep(15)
    else:
        raise RuntimeError('NCU guard not terminal before closure deadline; do not rewrite its status')
    # Terminal status can be written just before guard exit. Observe actual identity exit.
    guard=s.get('owned_guard',{})
    for _ in range(30):
        try:
            fields=Path(f"/proc/{guard['pid']}/stat").read_text().rsplit(')',1)[1].split()
            live=int(fields[19])==guard['birth_ticks'] and fields[0]!='Z'
        except (FileNotFoundError,KeyError):live=False
        if not live:break
        time.sleep(0.5)
    assert not live,'NCU guard still alive after terminal; preserve uncertainty'
    out.update(status='SUMMARIZING_TERMINAL',NCU_status=s['status']);save()
    run(['/usr/bin/python3',str(B/'summarize_ncu.py')])
    run(['/usr/bin/python3',str(B/'update_resource_report.py')])
    data=json.loads((N/'NCU-SUMMARY.json').read_text())
    report=B/'RESULT.md';t=report.read_text()
    t=t.replace('当前只剩本机NCU自然空闲窗口。','本机NCU窗口已终态，见[最终NCU](local-ncu-final-r2/RESULT.md)。')
    old='最新NCU首次窗口被外部进程打断，0kernel数据；新有限窗口等待，取得前相关细项UNKNOWN。'
    new='最新NCU首次窗口被外部进程打断，0kernel数据；最终窗口状态为'+data['status']+'，见[原始计数与边界](local-ncu-final-r2/RESULT.md)。未取得或不完整的细项仍UNKNOWN。'
    t=t.replace(old,new)
    t=t.replace('1. 完成最终版NCU后，将dynamic local、Tensor pipe和stall与上述stage表对应；无计数则保留UNKNOWN。','1. 审阅[最终版NCU](local-ncu-final-r2/RESULT.md)，将有效dynamic local、Tensor pipe和stall与上述stage表对应；未取得计数保留UNKNOWN。')
    report.write_text(t)
    run(['/usr/bin/python3',str(B/'update_hw.py')])
    selected=json.loads((B/'SELECTED.json').read_text())
    for f,h in selected['sources'].items():
        assert hashlib.sha256((R/'算子研究/wam_ops/optimized'/f).read_bytes()).hexdigest()==h, f
    head=run(['git','-C',str(M),'rev-parse','HEAD']).strip()
    remote=run(['git','-C',str(M),'ls-remote','origin','refs/heads/main']).split()[0]
    clean=not run(['git','-C',str(M),'status','--porcelain']).strip()
    if head!=job['expected_public_HEAD'] or remote!=head or not clean:
        out.update(status='LOCAL_REPORTS_COMPLETE_PUBLIC_CHANGED',head=head,remote=remote,mirror_clean=clean)
        save()
    else:
        run(['/usr/bin/python3',str(B/'export_review.py')])
        paths=json.loads((B/'PUBLIC-ALLOWLIST.json').read_text())
        for f in paths:
            if f.endswith('.json'):json.loads((M/f).read_text())
        for i in range(0,len(paths),70):run(['git','-C',str(M),'add','-f','--',*paths[i:i+70]])
        run(['git','-C',str(M),'diff','--cached','--check'])
        protected=['review/RESULTS.json','.pipeline/state.yaml','.pipeline/cycle.yaml','.pipeline/rules.yaml']
        assert not run(['git','-C',str(M),'diff','--cached','--name-only','--',*protected]).strip()
        msg=B/'CLOSURE-COMMIT-MESSAGE.txt';msg.write_text('Close bounded final-version NCU window and resource ledger\n\nPreserve the actual terminal state and all quality flags. Keep profiler counters separate from whole-layer benchmark timing.\n')
        run(['git','-C',str(M),'commit','-F',str(msg)])
        run(['git','-C',str(M),'push','origin','main'])
        new=run(['git','-C',str(M),'rev-parse','HEAD']).strip()
        assert run(['git','-C',str(M),'ls-remote','origin','refs/heads/main']).split()[0]==new
        out.update(status='LOCAL_AND_PUBLIC_CLOSURE_COMPLETE',commit=new,NCU_result=data['status'])
        save()
    # Mirror only final evidence under the authorized remote root; no model or job changes.
    remote_root='/root/cxz/ops/persistent-tile-20261019/evidence/async-gemm-20261020/overnight-alignment-20261022'
    send=[(str(p),remote_root+'/'+str(p.relative_to(B))) for p in list(N.glob('*.json'))+list(N.glob('*.md'))+list(N.glob('*.csv'))]
    send += [(str(B/f),remote_root+'/'+f) for f in ('RESULT.md','RESOURCE-AND-STOP-REPORT.json','RESOURCE-AND-STOP-REPORT.md')]
    transfer=B/'CLOSURE-REMOTE-ALLOWLIST.json';transfer.write_text(json.dumps(send)+'\n')
    run([sys.executable,'/tmp/send_wam_files.py',str(transfer)])
    out['remote_evidence_synced']=True
except Exception as e:
    out.update(status='CLOSURE_FAILED_OR_PARTIAL',error=repr(e),traceback=traceback.format_exc())
finally:
    out['end']=time.time();out['CPU_only_wall_lifecycle_seconds']=out['end']-out['start'];save()
