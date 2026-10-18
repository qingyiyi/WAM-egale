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

OUT=Path(__file__).resolve().parent


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
    lines=['# 限时性能采集：精度待验证','',f'更新：{result["at"]}。状态：{result["status"]}。','',
        f'实际{len(fixtures)}/54固定点，单进程、3轮；{len(rows)}性能单元、{events} Event样本及等量同步墙钟样本。attention每轮10样本，Graph每样本8调用；block每轮5样本。完整raw保留。','',
        '用户明确剩余独占不足20分钟，优先采集性能、输入输出后验。本协议独立命名；旧7轮/双进程协议的31完成点、用户收尾与全部消耗保留，不追溯改为通过。','',
        '**以下均为探索性延迟。启动前可见进程/显存为空，但NVML利用率非零原因未查明；未宣称物理宿主独占。精度、有效KV和全部状态契约待验证，不据此集成或签性能/质量证书。**','',
        '下表固定展示每档layer0/first query，其余全部层/query与慢轮见CSV；它不是挑选最快层。attention列为Graph-core，block为真实完整block eager。单位ms。','',
        '| 输入 | 实际self K | Torch self | standalone self | Torch block | standalone block | shared block | guarded block |',
        '|---|---|---|---|---|---|---|---|']
    for kind,batch in (('video',2),('action',2),('action',14)):
        for length in ('short','middle','long'):
            candidates=[p for p in data['points'] if (p['kind'],p['batch'],p['length'],p['layer'],p['query_role'])==(kind,batch,length,0,'first')]
            if not candidates:continue
            p=candidates[0];attention=p['attention']['self']['graph_core'];block=p['block']['eager']
            values=[attention['torch']['median_ms'],attention['standalone']['median_ms'],block['torch']['median_ms'],block['standalone']['median_ms'],block.get('shared',{}).get('median_ms'),block.get('guarded',{}).get('median_ms')]
            lines.append(f'| B{batch} {kind}/{length} | {p["attention"]["self"]["k_shape"][1]} | '+' | '.join('NA' if v is None else f'{v:.6f}' for v in values)+' |')
    lines+=['','## 时段与剩余工作','',f'阶段秒数（日志与文件时间边界）：`{json.dumps(phase,ensure_ascii=False)}`。性能段含边界资源核查，不将其算成纯kernel时间。','',
        '- 真实B14仍为7个独立Q16/CFG2，候选私有KV没有跨分支共享；video与action的实际K分别记录。',
        '- attention使用准备好的输入；B14 shared/guarded直接读取公共历史，完整block才包含其对应history/index/pack差异，二者边界不能混称。',
        '- 仅1个独立进程和3轮，接近噪声的微小优劣不能作为稳定收益；未完成原2进程长重复要求。',
        '- 原始与B14实际输入的代表性compact fixture已保存；后验先用它们验证单attention，完整block使用原冻结checkpoint，不复制大权重。',
        '- 精度/有效KV待验；未测30层完整verifier、接受翻转、scheduler、动作或机器人E2E。四项false保持。','',
        '[全部性能](PERFORMANCE-ONLY.csv) · [逐轮](PERFORMANCE-ONLY-ROUNDS.csv) · [输入形状](PERFORMANCE-ONLY-INPUTS.csv) · [机器摘要及fixture索引](PERFORMANCE-ONLY-SUMMARY.json)','',
        '原始证据：`'+str(job)+'`。','', 'CPU聚合问题：'+('无。' if not issues else repr(issues))]
    (OUT/'PERFORMANCE-ONLY-REPORT.md').write_text('\n'.join(lines)+'\n')
    (OUT/'CURRENT-STATUS.json').write_text(json.dumps({k:v for k,v in result.items() if k!='saved_inputs'},ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('status','points','performance_cells','event_samples','issues','phases')},ensure_ascii=False))


if __name__=='__main__':main()
