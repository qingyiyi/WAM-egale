"""只汇总已完成原始记录；保留A，绝不把负结果改为通过。"""
from pathlib import Path
import datetime
import hashlib
import json
import runpy
import statistics

P=Path(__file__).resolve().parent
S=runpy.run_path(str(P/'stage_report.py'))['STRUCTURE']
local=json.loads((P/'paired-local-comparison/data/RESULT.json').read_text())
assert local['status']=='COMPLETE_FINITE_DIAGNOSTIC'
measure=[json.loads((P/f'T2-r1/results/measure{i}-data/RESULT.json').read_text()) for i in [1,2]]
assert all(d['status']=='COMPLETE_EXACT_EAGER' for d in measure)
precision=json.loads((P/'T2-r1/results/precision/RESULT.json').read_text())
assert precision['status']=='FINITE_EXACT_PASS'
now=datetime.datetime.now().astimezone().isoformat()


def q(values,p):
    a=sorted(values);x=(len(a)-1)*p;i=int(x)
    return a[i] if i==len(a)-1 else a[i]+(x-i)*(a[i+1]-a[i])


normal=[]
for proc,d in enumerate(measure,1):
    for row in d['rows']:
        grouped={}
        for v in row['samples']:grouped.setdefault(v['rep'],{})[v['arm']]=v['event_ms']
        ratios=[v['A']/v['Own'] for v in grouped.values()]
        x=dict(process=proc,case=row['case'],A_ms=row['median_ms']['A'],candidate_ms=row['median_ms']['Own'],Torch_ms=row['median_ms']['Torch'],Guarded_ms=row['median_ms']['Guarded'],A_over_candidate=row['median_ms']['A']/row['median_ms']['Own'],ratio_P10=q(ratios,.1),ratio_P90=q(ratios,.9),candidate_resources=row['resources'])
        x['regression_percent']=(x['candidate_ms']/x['A_ms']-1)*100
        assert x['regression_percent']>5
        normal.append(x)
for row in measure[0]['rows']:
    assert len(row['seven_segments'])==7
    assert all(v['partial']['bitwise_pass'] and v['running']['bitwise_pass'] for v in row['seven_segments'])
ffn=[dict(case=r['case'],A_ms=r['region_medians_ms']['A'][15],candidate_ms=r['region_medians_ms']['PAIR'][15],A_over_candidate=r['ffn_speed_A_over_candidate']['PAIR']) for r in local['rows']]
result=dict(status='CLOSED_NO_GO_RETAIN_A',classification='NO_GO',at=now,design='PAIR2048x2',retained='SELF-QKV-TMA-4MMA',adopted=False,finite_precision='PASS: three actual block0 fixtures, seven P_s and R_s, captured intermediates/output and input/KV/RNG',normal_whole_block=normal,same_round_FFN_profile=ffn,new_designs=1,preparation_revision=1,engineering_quality_screen=False,independent_test_complete=False,noninferiority_established=False,formal_performance_permission=False,production_admitted=False)
(P/'RESULTS.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')

sources={}
for relative in ['A-source/csrc/whole_layer.cu','A.so','T2-r1/whole_layer.cu',*map(lambda f:str(f.relative_to(P)),(P/'T2-r1/build/normal').glob('*.so')),*map(lambda f:str(f.relative_to(P)),(P/'T2-r1/build/profile').glob('*.so'))]:
    f=P/relative;sources[relative]=dict(bytes=f.stat().st_size,sha256=hashlib.sha256(f.read_bytes()).hexdigest())
(P/'FINAL-SOURCE-MANIFEST.json').write_text(json.dumps(sources,ensure_ascii=False,indent=2)+'\n')


def stage(task):
    lines=[f'# {task}：实际保留A完整阶段表','',f'时间{now}；评估PAIR2048x2；结论NO_GO；实际保留SELF-QKV-TMA-4MMA，原优化入口未改。',
        'RTX5090/SM120 GPU-e5e7821f-9633-6cc8-d06c-d0d0df4c9256；B14/Q16/block0，KV288/1152/2160；warm CrossKV；eager/no Graph。',
        'A源码/独立二进制与候选的完整身份在FINAL-SOURCE-MANIFEST.json；A的源码/DSO SHA256分别为'+sources['A-source/csrc/whole_layer.cu']['sha256']+' / '+sources['A.so']['sha256']+'。',
        '正常A/候选同资源：170CTA×256线程，1CTA/SM，shared100352B、255reg、静态local784B；profile静态local792B。动态spill/DRAM/L2计数UNKNOWN。','',
        '|阶段|形状/tile|CTA任务/owner|warp|ACT/partial/acc存储|同步|ready/free/提交|','|---|---|---|---|---|---|---|']
    for i,row in enumerate(S,1):lines.append('|'+str(i)+'. '+'|'.join(row)+'|')
    lines += ['', '以上主结构描述实际保留A。被拒候选只改第16行：96输出tile×4组=384 owner任务，仍672段MMA；组内R0/R2/R4 shared，组间R1/R3/R5 global，七次舍入；group边界release 2/4/6/7。FFN1重算1×，ACT仍global。',
        'FFN1主M128/N256/K_step32，尾M32/N256/K_step64；FFN2 M64/N128，数值K2048、内部K_step64。原ACT四生产组发布，候选不改。候选compute73728B+BF16 running16384B=90112B，小于现有TileShared98304B。',
        'warm Cross K/V投影与K RMS为NOT_EXECUTED；层外预计算另列。','',
        '|阶段|A μs 短/中/长|原Torch μs 短/中/长|Torch/A|被拒PAIR μs 短/中/长|A/PAIR|状态/来源|','|---|---:|---:|---:|---:|---:|---|']
    fmt=lambda xs:'/'.join(f'{v:.3f}' for v in xs)
    for i,row in enumerate(S):
        a=[r['region_medians_ms']['A'][i]*1000 for r in local['rows']]
        b=[r['region_medians_ms']['PAIR'][i]*1000 for r in local['rows']]
        t=[r['torch_region_ms'].get(str(i)) for r in local['rows']]
        tr=fmt([v*1000 for v in t]) if all(v is not None for v in t) else 'NA'
        ta='/'.join(f'{x*1000/y:.3f}×' for x,y in zip(t,a)) if tr!='NA' else 'NA'
        ab='/'.join(f'{x/y:.3f}×' for x,y in zip(a,b))
        lines.append(f'|{i+1}. {row[0]}|{fmt(a)}|{tr}|{ta}|{fmt(b)}|{ab}|CURRENT_MEASURED；paired-local-comparison|')
    lines += ['', '阶段为两份独立profile DSO的嵌入区间，原Torch是同job独立eager数学区域；含同步/重叠，不能相加代替整层。完整block采用判定只用以下无插桩原始记录。','',
        '|进程/KV|A μs|PAIR μs|A/PAIR|配对比值P10–P90|PAIR退化|正确性/决定|','|---|---:|---:|---:|---|---:|---|']
    for r in normal:
        lines.append(f'|{r["process"]}/{r["case"]}|{r["A_ms"]*1000:.3f}|{r["candidate_ms"]*1000:.3f}|{r["A_over_candidate"]:.4f}×|{r["ratio_P10"]:.4f}–{r["ratio_P90"]:.4f}|+{r["regression_percent"]:.2f}%|有限逐位PASS；NO_GO，保留A|')
    lines += ['', 'P10–P90为16个交错轮次的描述性波动区间，不是置信区间。全量样本保存在RESULT.json，未挑最快样本。','',
        '|KV|当前同轮原Torch/A|当前同轮Guarded/A|FFN A/PAIR（同轮profile）|','|---|---|---|---|']
    for name in ['short','middle','long']:
        ns=[v for v in normal if v['case']==name];f=next(x for x in ffn if x['case']==name)
        lines.append('|'+name+'|'+'/'.join(f'{x["Torch_ms"]/x["A_ms"]:.4f}×' for x in ns)+'|'+'/'.join(f'{x["Guarded_ms"]/x["A_ms"]:.4f}×' for x in ns)+f'|{f["A_over_candidate"]:.4f}×|')
    lines += ['', '有限正确性范围是三份真实block0输入；其他层/在线接受/完整模型/闭环质量、Graph replay和B2 fallback未扩验，不借用本结果放行。旧Graph与旧A100结果未混入。']
    text='\n'.join(lines)+'\n';(P/'snapshots'/f'{task}-STAGE-TABLE.md').write_text(text);(P/'STAGE-TABLE.md').write_text(text)
    return text


stage('T2-NO-GO')
review=f'''# SM120 FFN接续：NO_GO，保留当前A

OBSERVED：T0/T1/T2完成一个PAIR2048x2设计。相邻两段同CTA，组内BF16状态shared交接，七段原FP32 partial与BF16 running逐段有限逐位通过；原捕获中间量、输出、输入/有效KV/RNG和新384-owner schema通过。没有改容差、GELU、bias/残差、模型、权重或ACT。

正常整层两独立进程在所有三KV均慢6.25%–7.89%；同轮FFN profile候选/A约1.18–1.19倍。结论NO_GO，当前SELF-QKV-TMA-4MMA源码和二进制保持不变。不是数值失败，不晋级。

## 可复核事实

- A/候选正常资源完全同为170CTA/1CTA每SM/shared100352B/reg255/static-local784B。不能把退化归因为扩大shared或静态资源导致occupancy下降。
- 72KiB compute union加16KiB BF16状态可放进现有96KiB TileShared，没有压小M或重复FFN1。候选确实保留ACT和七次舍入，仅将672段任务分为384 owner组。
- T1前序等待摊分仅0.20–0.22μs/任务，MMA与输入/结果交接包络25.8μs，提交转换1.9μs、发布0.43μs。不是纯Tensor Core时间，也不是全GPU关键路径比例。
- 最大可省7.875MiB是源码逻辑global往返，增加同量shared交接；没有DRAM/L2实测，不写HBM节省。

HYPOTHESIS：任务变粗导致split-K负载均衡/尾波变差，与退化方向一致。但新提交逻辑和任务归属同时改变，没有逐tile硬件trace或受控分离，不能把全部退化直接归因该因素。既有静态资源不变排除了“单纯多占shared”的解释，不能排除动态spill/cache等影响。

UNKNOWN：DRAM/L2分项、真实动态spill、critical CTA任务分布、精确Tensor Core忙碌率。软件包络不是NCU；没有为已明确负收益再编译新的诊断候选。

## 用户五个问题

1. 当前最合理的保留配置是已验证A：FFN1主M128/N256/K32，尾M32/N256/K64；FFN2 M64/N128、七个K2048、内部K64。它是在已测方案中更合适，不是全局最优证明；更细ACT ready粒度尚未验证。
2. 通过global ACT让24个输出列tile共享一次FFN1，重算1×；代价是ACT存储/读取与ready发布。此前M16/DSM无重算仍慢，本轮不重跑。
3. 保留FFN1原cp.async/MMA→bias/BF16→GELU/BF16→ACT global→release ready；FFN2原四warp MMA→shared FP32 result→八warp逐段合并/转换/写回→fence/release。六投影的专职TMA路径与本轮FFN是不同子图，未改。
4. 七段逐次舍入完整保留。s0=B(P0)，s1..5=B(Ps+FP32(Rprev))，s6=B((P6+FP32(R5))+b2)，最终再按原门控残差顺序舍入。每段P都独立清零，不把上一段R作为MMA初值。三真实fixture的全部七段原始P/R已保存比较；不外推全部输入。
5. 保留global ACT和当前全局partial方案，因为本轮省partial逻辑量未转化收益，驻留旧方案也无收益。没有证据支持现在强行删除ACT或新调warp比例。

## 决定与下一步

停止PAIR2048x2，不做第二设计或tile/warp搜索。建议下一轮优先解释FFN2约25.8μs的MMA/输入交接包络，或单独讨论更细ACT ready边界；两者不在本轮自动展开。允许答案是当前global ACT版本暂时最合适。

完整表STAGE-TABLE.md；机器结果RESULTS.json；原始正常数据T2-r1/results/measure{{1,2}}-data/RESULT.json；同轮局部paired-local-comparison/data/RESULT.json；T1数值原件T1-r1/data/numeric-*.pt；候选数值T2-r1/results/measure1-data/numeric-candidate-*.pt；独立A.so、A-source、T2-r1/CHANGE.patch及FINAL-SOURCE-MANIFEST.json固定源码/二进制身份。

本机与远端科研根不是Git checkout，不能伪造HEAD；原始来源为已发布3eb0026705e603d6268731bb26b34261acca696c及现场字节一致的采用记录，当前固定文件身份优先。公开main镜像提交另存publication收据，不以远端main覆盖判断。

engineering_quality_screen=false；independent_test_complete=false；noninferiority_established=false；formal_performance_permission=false；production_admitted=false。这里只是有限单层真实重放，不是在线verifier或机器人E2E。
'''
(P/'REVIEW_RETURN.md').write_text(review)
(P/'CLAIM-EVIDENCE-MATRIX.md').write_text('''# 本轮主张与证据

|主张|已有证据|缺口|允许写入|
|---|---|---|---|
|七段BF16状态可在相邻段同owner时保持|三fixture每段P/R及27捕获项有限逐位|未覆盖其他输入/层/在线路径|仅有限实现正确性|
|省逻辑partial global读写可直接加速|正常两进程反而慢6.25%–7.89%|硬件流量/细机制未分离|不允许正收益主张|
|退化因shared增大/occupancy下降|正常资源完全不变|实际动态指标未知|不允许此归因|
|退化因split-K并行减小|672→384 owner且序列段变长|无单因素分离与critical CTA trace|仅假设|
|当前A比本轮候选更快|同轮两进程三KV完整block/FFN，保留全部样本|不是所有硬件/输入最优|允许限定主张|
|机器人质量或E2E已放行|本轮未测|原四科学门false|不允许|
''')
entry=dict(retained='SELF-QKV-TMA-4MMA',decision='NO_GO_PAIR2048x2',live_source='/rjs/cxz/论文探索/.pipeline/experiments/rtx5090-block-port-20261023/workspace/算子研究/wam_ops/optimized/csrc/whole_layer.cu',frozen_A='A-source/csrc/whole_layer.cu',independent_A_binary='A.so',rejected_candidate='T2-r1/whole_layer.cu',patch='T2-r1/CHANGE.patch',normal_raw=['T2-r1/results/measure1-data/RESULT.json','T2-r1/results/measure2-data/RESULT.json'],local_raw='paired-local-comparison/data/RESULT.json',manifest='FINAL-SOURCE-MANIFEST.json',adopted=False,all_scientific_flags=False)
(P/'SOURCE-AND-RESULT-ENTRY.json').write_text(json.dumps(entry,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(dict(status=result['status'],normal=normal,FFN=ffn),indent=2))
