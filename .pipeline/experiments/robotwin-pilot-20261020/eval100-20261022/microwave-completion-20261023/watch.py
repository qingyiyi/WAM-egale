"""Wait for the fixed 100 microwave triplets; only read existing results."""
from pathlib import Path
import datetime
import json
import os
import resource
import signal
import time
import traceback

J = Path(__file__).resolve().parent
if (J / 'STATUS.json').exists():
    raise SystemExit('Existing watch attempt must not be overwritten')
P = json.loads((J / 'JOB.json').read_text())
D = Path(P['queue_root'])
deadline = datetime.datetime.fromisoformat(P['deadline']).timestamp()
os.sched_setaffinity(0, {18})
resource.setrlimit(resource.RLIMIT_CPU, (600, 610))
resource.setrlimit(resource.RLIMIT_FSIZE, (16 * 1024**2, 16 * 1024**2))

def save(name, value):
    p = J / name
    tmp = p.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(p)

def stamp():
    return datetime.datetime.now().astimezone().isoformat()

def stop(signum, _frame):
    raise SystemExit(f'control_signal_{signum}')

signal.signal(signal.SIGTERM, stop)
signal.signal(signal.SIGINT, stop)
signal.signal(signal.SIGXCPU, stop)
birth = int(Path(f'/proc/{os.getpid()}/stat').read_text().rsplit(')', 1)[1].split()[19])
s = dict(status='WAITING_COMPLETE_100_MICROWAVE_PAIRS', start=stamp(), pid=os.getpid(),
         birth_ticks=birth, polls=0, GPU_used=False, deadline=P['deadline'])
try:
    save('STATUS.json', s)
    while time.time() < deadline:
        q = json.loads((D / 'PAIR-CLAIMS.json').read_text())
        pairs = [p for p in q['pairs'].values() if p['status'] == 'COMPLETE_PAIR'
                 and any(e.get('task') == 'open_microwave' for e in p.get('episodes', []))]
        es = [e for p in pairs for e in p['episodes']]
        errors = []
        ids = []
        for p in pairs:
            rows = p['episodes']
            indices = {e['valid_index'] for e in rows}
            if (len(rows) != 3 or {e['arm'] for e in rows} != {'T', 'C', 'P'}
                    or len(indices) != 1 or any(e['task'] != 'open_microwave' for e in rows)):
                errors.append('invalid triplet identity')
                continue
            ids.extend(indices)
            for e in rows:
                expected = 'COMPLETE_SUCCESS' if e['success'] else 'COMPLETE_ALGORITHMIC_FAILURE'
                if e['status'] != expected:
                    errors.append('non-natural terminal included')
        if len(ids) != len(set(ids)) or any(i not in range(100) for i in ids):
            errors.append('duplicate or out-of-manifest index')
        counts = {a: dict(complete=sum(e['arm'] == a for e in es),
                          success=sum(e['arm'] == a and bool(e['success']) for e in es))
                  for a in ['T', 'C', 'P']}
        s.update(last_check=stamp(), polls=s['polls'] + 1, complete_pairs=len(pairs),
                 counts=counts, identity_errors=errors, global_halt=q.get('halt'))
        save('STATUS.json', s)
        if errors:
            s['status'] = 'BLOCKED_IDENTITY_REVIEW_REQUIRED'
            break
        if set(ids) == set(range(100)):
            audit_ok = all(p.get('prefix_audit', {}).get('pass_all') is True for p in pairs)
            evidence = []
            for e in es:
                r = Path(e['namespace']) / e['episode_id']
                evidence.append(dict(episode=e, directory=str(r),
                                     present_files=sorted(x.name for x in r.iterdir()) if r.exists() else [],
                                     provenance='existing registered episode; no replay'))
            save('COMPLETE-PANEL.json', dict(asof=stamp(), source=str(D / 'PAIR-CLAIMS.json'),
                 counts=counts, pair_audits_passed=audit_ok, episodes=evidence,
                 engineering_quality_screen=False, independent_test_complete=False,
                 noninferiority_established=False, formal_performance_permission=False))
            (J / 'DIAGNOSIS-READY.md').write_text(
                '# 微波炉300局已完整，等待原因核查\n\n'
                '原始逐局终态、命令计数与产物路径见COMPLETE-PANEL.json。\n'
                f'100组身份核对完成；原prefix审计全部通过：{audit_ok}。\n\n'
                '先核原Target官方复现差距，再分析TF/PF2额外失败。\n'
                '本脚本只整理证据；没有确定因果根因，没有启动GPU或改变科学权限。\n')
            s['status'] = 'COMPLETE_PANEL_READY_FOR_DIAGNOSIS' if audit_ok else 'COMPLETE_PANEL_AUDIT_REVIEW_REQUIRED'
            break
        if q.get('halt'):
            s['status'] = 'PARTIAL_CAMPAIGN_HALTED'
            break
        worker_files = list(D.glob('gpu*/STATUS.json'))
        workers = [json.loads(p.read_text()) for p in worker_files]
        if workers and all(w.get('end') for w in workers):
            s['status'] = 'PARTIAL_WORKERS_TERMINAL_BEFORE_100'
            break
        time.sleep(min(300, max(0, deadline - time.time())))
    else:
        s['status'] = 'PARTIAL_ORIGINAL_DEADLINE_REACHED'
except SystemExit as e:
    s.update(status='STOPPED_CONTROL_SIGNAL', error=str(e))
except BaseException as e:
    s.update(status='FAILED_CPU_WATCH_NO_RETRY', error=repr(e), traceback=traceback.format_exc())
finally:
    s.update(end=stamp(), process_cpu_seconds=time.process_time(), owned_children=[],
             cleanup='No child or GPU processes; this watch exits after writing terminal status')
    save('STATUS.json', s)
