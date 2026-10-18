"""Aggregate the separately registered short-window performance capture on CPU."""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import csv
import json
import math
import os
import statistics
import sys

OUT=Path(__file__).resolve().parent/'performance'


def load(path):return json.loads(path.read_text())


def write_csv(name,rows):
    with (OUT/name).open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        w.writeheader();w.writerows(rows)


def main():
    os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
    job=Path(sys.argv[1]).resolve(strict=True);path=job/'process1/RESULT.json'
    if not path.exists():
        result=dict(status='PERFORMANCE_CAPTURE_NOT_OBTAINED',job=str(job),input_output_validation='PENDING',
            terminal=load(job/'TERMINAL.json') if (job/'TERMINAL.json').exists() else None)
        (OUT/'PERFORMANCE-ONLY-SUMMARY.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
        print(json.dumps(result,ensure_ascii=False));return
    data=load(path);issues=[];rows=[];rounds=[];fixtures=[];events=walls=0
    def require(ok,what):
        if not ok:issues.append(what)
    for index,point in enumerate(data['points']):
        keys={k:point[k] for k in ('length','batch','kind','query_role','query_index','layer')}
        keys.update(source_pointer=f'{path}#/points/{index}',input_output_validation='PENDING')
        self_data=point['attention']['self']
        fixtures.append(dict(**keys,q_shape=str(self_data['q_shape']),self_k_shape=str(self_data['k_shape']),
            cross_k_shape=str(point['attention']['cross']['k_shape']),dtype=self_data['dtype'],prefix_tokens=point['prefix_tokens']))
        for part,section in (('self',point['attention']['self']),('cross',point['attention']['cross']),('block',point['block'])):
            for mode in ('eager',) if part=='block' else ('eager','graph_core'):
                measured=section[mode];base=measured['torch']['median_ms'];shared=measured.get('shared',{}).get('median_ms')
                for engine,item in measured.items():
                    raw=item['raw_event_ms'];wall=item['raw_wall_ms'];n=5 if part=='block' else 10
                    require(len(raw)==len(wall)==3,f'{keys}/{part}/{mode}/{engine}:rounds')
                    require(all(len(v)==n for v in raw+wall),f'{keys}/{part}/{mode}/{engine}:samples')
                    med=[statistics.median(v) for v in raw];value=statistics.median(med)
                    require(math.isclose(value,item['median_ms'],rel_tol=1e-12),f'{keys}/{part}/{mode}/{engine}:median')
                    require(all(math.isfinite(x) and x>0 for r in raw+wall for x in r),f'{keys}/{part}/{mode}/{engine}:finite-positive')
                    events+=sum(map(len,raw));walls+=sum(map(len,wall))
                    rows.append(dict(**keys,part=part,mode=mode,engine=engine,engine_ms=value,torch_ms=base,
                        engine_over_torch=value/base,torch_over_engine=base/value,
                        shared_ms=shared,engine_over_shared=value/shared if shared else None,
                        synchronized_wall_ms=statistics.median(statistics.median(v) for v in wall),
                        rounds=3,samples_per_round=n,graph_calls_per_sample=8 if mode=='graph_core' else 1,
                        min_round_ms=min(med),max_round_ms=max(med),numerical_gate_passed='NOT_EVALUATED'))
                    for rid,v in enumerate(med):rounds.append(dict(**keys,part=part,mode=mode,engine=engine,round=rid+1,median_ms=v))
    terminal=load(job/'TERMINAL.json') if (job/'TERMINAL.json').exists() else None
    require(len(data['points'])==54,'54 fixed point coverage incomplete')
    require(data['status']=='COMPLETE_PERFORMANCE_CAPTURE_VALIDATION_DEFERRED','worker incomplete')
    require(terminal is not None and terminal['owned_children_exited'],'cleanup not observed')
    saved=load(job/'process1/SAVED-INPUTS.json') if (job/'process1/SAVED-INPUTS.json').exists() else None
    phase={}
    owned=job/'process1/OWNED-PROCESS.json'
    if owned.exists():
        phase['process_start_unix']=owned.stat().st_mtime
        for line in (job/'process1/worker.log').read_text(errors='replace').splitlines():
            if 'Component devices: transformer=' in line:
                ready=datetime.strptime(line[:23],'%Y-%m-%d %H:%M:%S,%f').replace(tzinfo=ZoneInfo('Asia/Shanghai')).timestamp()
                phase['model_load_and_initialization_seconds']=ready-phase['process_start_unix']
                if saved:
                    prepared=(job/'process1/SAVED-INPUTS.json').stat().st_mtime
                    phase['real_input_reconstruction_and_compact_save_seconds']=prepared-ready
                    phase['performance_and_resource_checks_seconds']=path.stat().st_mtime-prepared
                break
    if terminal:phase['whole_GPU_job_lifecycle_seconds']=terminal['wall_seconds']
    result=dict(status='COMPLETE_PERFORMANCE_CAPTURE_VALIDATION_PENDING' if not issues else 'PARTIAL_PERFORMANCE_CAPTURE',
        at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds'),job=str(job),issues=issues,points=len(fixtures),
        independent_processes=1,rounds=3,performance_cells=len(rows),round_cells=len(rounds),event_samples=events,wall_samples=walls,
        input_output_validation='PENDING',latency_resource_status='EXPLORATORY_INITIAL_NVML_COUNTER_UNQUALIFIED',
        terminal=terminal,saved_inputs=saved,phases=phase,
        engineering_quality_screen=False,independent_test_complete=False,noninferiority_established=False,formal_performance_permission=False,
        supersedes_execution_order_only=True,old_long_repetition_protocol='PARTIAL_PRESERVED_NOT_REWRITTEN')
    write_csv('PERFORMANCE-ONLY.csv',rows);write_csv('PERFORMANCE-ONLY-ROUNDS.csv',rounds);write_csv('PERFORMANCE-ONLY-INPUTS.csv',fixtures)
    (OUT/'PERFORMANCE-ONLY-SUMMARY.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    lines=['# 新v2性能复测','',f'状态：{result["status"]}。54固定点，{len(rows)}单元，{events} Event和等量同步wall样本。',
        '', '此处standalone字段指已去fast_math的新v2。与M21旧二进制时间分开，不借旧结果。',
        '', '精度依据M22先前同代码的108attention/54block/六full30/三完整policy逐位验证；本计时过程不在计时区间重复误差统计。新捕获输入的CPU同源核对另列。',
        '', '启动时无可见他人context/0MiB，用户确认GPU空闲；空载NVML25%异常仍未解释，单进程3轮，保留全部慢点，只作限定资源条件下的延迟证据。',
        '', '覆盖B2 video/action、B14独立七候选CFG2，各三KV×first/last×层0/8/29。kernel eager/Graph-core、完整block eager。不是完整机器人E2E或新MegaTransformer收益。',
        '', '[全部性能](PERFORMANCE-ONLY.csv) · [逐轮](PERFORMANCE-ONLY-ROUNDS.csv) · [输入](PERFORMANCE-ONLY-INPUTS.csv)',
        '', '聚合问题：'+repr(issues), '', '原始：'+str(job)]
    (OUT/'PERFORMANCE-ONLY-REPORT.md').write_text('\n'.join(lines)+'\n')
    (OUT/'CURRENT-STATUS.json').write_text(json.dumps({k:v for k,v in result.items() if k!='saved_inputs'},ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('status','points','performance_cells','event_samples','issues','phases')},ensure_ascii=False))


if __name__=='__main__':main()
