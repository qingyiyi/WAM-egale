"""Preserve per-kernel metrics; never use profiler latency as benchmark latency."""
from pathlib import Path
import csv
import io
import json

B=Path(__file__).resolve().parent
N=B/'local-ncu-final-r2'
s=json.loads((N/'STATUS.json').read_text())
terminal=s['status'] in ('COMPLETE','FAILED_OR_PARTIAL','EXPIRED_WITHOUT_GPU_WINDOW')
assert terminal, 'NCU attempt is not terminal'
worker=json.loads((N/'RESULT.json').read_text()) if (N/'RESULT.json').exists() else {}
text=(N/'NCU.csv').read_text() if (N/'NCU.csv').exists() else ''
lines=text.splitlines()
header=next((i for i,l in enumerate(lines) if l.startswith('"ID","Process ID"')),None)
kernels={}
if header is not None:
    for r in csv.DictReader(io.StringIO('\n'.join(lines[header:]))):
        if not r.get('Metric Name') or not r.get('Metric Value'):
            continue
        key=(r['ID'],r['Kernel Name'])
        k=kernels.setdefault(key,dict(id=r['ID'],name=r['Kernel Name'],block=r['Block Size'],grid=r['Grid Size'],metrics={}))
        raw=r['Metric Value']
        try:value=float(raw.replace(',',''))
        except ValueError:value=raw
        k['metrics'][r['Metric Name']]=dict(value=value,unit=r['Metric Unit'])
accepted=s['status']=='COMPLETE' and worker.get('status')=='COMPLETE_NCU_FINITE_REAL_INPUT' and worker.get('profiled_calls')==4 and bool(kernels)
data=dict(status='OBSERVED_FINAL_VERSION_COUNTERS' if accepted else 'COUNTERS_UNKNOWN_OR_PARTIAL',
          guard_status=s['status'],worker_status=worker.get('status','NOT_PRODUCED'),
          module_sha256=json.loads((N/'JOB.json').read_text())['module_sha256'],
          kernels=list(kernels.values()),accepted_for_diagnostic=accepted,
          hardware_counter_snapshot_only=True,formal_performance=False,
          limitations=['No clock/cache controls','Kernel replay counts not independently logged',
                       'Native ranges are individual stages, not a complete native layer',
                       'Local-path bytes are not necessarily DRAM traffic',
                       'Stall percentages are warp issue statistics, not wall-time fractions'])
(N/'NCU-SUMMARY.json').write_text(json.dumps(data,indent=2)+'\n')
out=['# 最终版本本机NCU', '', f"状态：{data['status']}；guard={s['status']}，worker={data['worker_status']}。", '',
     '真实middle/B14/action/layer0，冻结DSOac3f9e46；仅诊断，不用于正式延迟表。先前0kernel外部干扰记录保持PARTIAL。', '']
if accepted:
    out += ['OBSERVED：四个NVTX范围完成；每个实际kernel分别列出，不将多kernel的百分比直接相加。', '',
            '|kernel|grid/block|Tensor active %|active warp %|DRAM bytes|L2 sectors|local load+store bytes|barrier stall %|',
            '|---|---|---:|---:|---:|---:|---:|---:|']
    for k in kernels.values():
        d=k['metrics'];v=lambda key:d.get(key,{}).get('value','UNKNOWN')
        ld=v('l1tex__t_bytes_pipe_lsu_mem_local_op_ld.sum');st=v('l1tex__t_bytes_pipe_lsu_mem_local_op_st.sum')
        local=ld+st if isinstance(ld,(int,float)) and isinstance(st,(int,float)) else 'UNKNOWN'
        out.append(f"|{k['name']}|{k['grid']} / {k['block']}|{v('sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active')}|{v('sm__warps_active.avg.pct_of_peak_sustained_active')}|{v('dram__bytes.sum')}|{v('lts__t_sectors.sum')}|{local}|{v('smsp__warp_issue_stalled_barrier_per_warp_active.pct')}|")
else:
    out += ['UNKNOWN：未满足无外部干扰的完整counter采集要求，不用部分数据/旧版本计数替代。完整层性能与精度结果独立成立，本次缺口保留。']
out += ['', 'HYPOTHESIS：local状态保存/恢复、固定资源配置和执行几何可能共同限制整层；这些counter不能单独确定每项耗时。', '',
        'RECOMMENDATION：结合三KV逐阶段和资源干预解释剩余差距。没有完整native层counter，不宣称整层流量相对native下降或增加某倍数；不使用NCU时间生成speedup。', '']
(N/'RESULT.md').write_text('\n'.join(out))
print(data['status'],len(kernels))
