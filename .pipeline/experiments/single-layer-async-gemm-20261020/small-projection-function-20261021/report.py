from pathlib import Path
import json,csv,statistics,datetime,shutil
R=Path('/rjs/cxz/论文探索')
B=R/'.pipeline/experiments/single-layer-async-gemm-20261020/small-projection-function-20261021'
old=json.loads((B.parent/'remaining-gemm-audit-20261021/STEADY-ANALYSIS.json').read_text())
new=json.loads((B/'stage-control/ANALYSIS.json').read_text())
now=datetime.datetime.now().astimezone().isoformat()
six=['self_Q','self_K','self_V','self_out','cross_Q','cross_out']
merged=[]
for n in new['rows']:
 if n['condition']!='same_real_native_prefix':continue
 o=next(x for x in old['rows'] if all(x[k]==n[k] for k in ['process','case','name','condition']))
 merged.append(dict(process=n['process'],case=n['case'],name=n['name'],old_native_us=o['native_us'],new_native_us=n['native_us'],old_independent_marker_us=o['own_us'],new_independent_marker_us=n['own_us'],old_independent_inside_us=o['inside_us'],new_independent_inside_us=n['inside_us'],old_embedded_us=o['embedded_us'],new_embedded_us=n['embedded_us'],embedded_change_us=n['embedded_us']-o['embedded_us']))
with (B/'STAGE-BEFORE-AFTER.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(merged[0]),lineterminator='\n');w.writeheader();w.writerows(merged)
def span(a,key):return f"{min(x[key] for x in a):.3f}–{max(x[key] for x in a):.3f}"
stage_table='|阶段|旧独立调用体 µs|新独立调用体 µs|旧整层内 µs|新整层内 µs|\n|---|---:|---:|---:|---:|\n'
for name in six:
 a=[x for x in merged if x['name']==name]
 stage_table+='|'+name+'|'+'|'.join(span(a,k) for k in ['old_independent_inside_us','new_independent_inside_us','old_embedded_us','new_embedded_us'])+'|\n'
statuses=[json.loads((p/'STATUS.json').read_text()) for p in [B,B/'stage-control']]
job=json.loads((B/'JOB.json').read_text());jobs=[j for s in statuses for j in s['jobs']]
gpu=sum(j['lifecycle_seconds'] for j in jobs if j['gpu']);cpu=sum(j['lifecycle_seconds'] for j in jobs if not j['gpu'])
precision=[]
for proc in [1,2]:
 d=json.loads((B/f'stage-control/RESULT-{proc}.json').read_text())
 assert d['status']=='COMPLETE_CURRENT_SOURCE_PREFIX_DIAGNOSTIC'
 for row in d['rows']:
  assert row['cache_check']['pass_all'] and row['inputs_unchanged'] and row['RNG_unchanged']
  for s in row['stages']:
   for c in s['conditions']:
    precision.extend(c['precision'].values())
assert len(precision)==240 and all(p['bitwise_pass'] and p['max_abs']==0 for p in precision)
clocks=[{k:x[k] for k in ['process','case','name','condition','clock_before','clock_after']} for x in new['rows'] if any(x[p]['SM']!=1410 for p in ['clock_before','clock_after'])]
ps=dict(finite_gate_signatures=9,gate_full_block_calls=72,common_intermediates_per_signature=23,valid_KV_transitions=18,stage_comparisons=len(precision),stage_max_abs=0,stage_case_state_checks=6,gate='SIMPLE-GATE-RESULT.json',stage_raw=['stage-control/RESULT-1.json','stage-control/RESULT-2.json'],not_general_proof=True,production_admitted=False)
(B/'PRECISION-SUMMARY.json').write_text(json.dumps(ps,indent=2)+'\n')
res=dict(status='COMPLETE_RETAINED_PARTIAL_IMPROVEMENT',observed_at=now,GPU_lifecycle_seconds=gpu,CPU_prepare_build_lifecycle_seconds=cpu,CPU_disassembly_seconds=json.loads((B/'SASS-STATIC-SUMMARY.json').read_text())['CPU_seconds'],CUDA_active_seconds='UNKNOWN',cumulative_prior_GPU_seconds=job['cumulative_prior_GPU_seconds'],cumulative_GPU_seconds=job['cumulative_prior_GPU_seconds']+gpu,cumulative_cap_seconds=36000,remaining_cumulative_GPU_seconds=36000-job['cumulative_prior_GPU_seconds']-gpu,phase_GPU_cap_seconds=1800,deadline=job['deadline'],failures=[],CPU_report_errors=[dict(error='KeyError candidate',reason='report used candidate key; raw timing uses specialized',resolution='corrected derived report reader; raw evidence unchanged; no GPU rerun')],timeouts=[],new_candidates=1,stage_clock_exceptions=clocks,cleanup='CLEANUP-CHECK.json',owned_cleanup_pass=True,stop_reason='One registered candidate completed; no additional candidate, GPU rerun, DAG, quality or robot task started',limitations=['Self-Q slower; partial resolution only','New independent small projections slower than previous independent implementation','Stage before/after from different processes and sessions; strongest causal evidence is paired full-layer old/new timing','Three hot-only context clock samples at 1395MHz preserved; all same-prefix primary samples and embedded boundaries 1410MHz'],four_scientific_permissions_remain_false=True)
(B/'RESOURCE-AND-STOP-REPORT.json').write_text(json.dumps(res,ensure_ascii=False,indent=2)+'\n')
timings=json.loads((B/'WHOLE-LAYER-COMPARISON.json').read_text())
print('whole timing schema',type(timings),timings[0] if isinstance(timings,list) else list(timings))
whole='|KV|进程1：旧→新 ms|进程2：旧→新 ms|耗时下降|\n|---|---:|---:|---:|\n'
for case,kv in [('short',288),('middle',1152),('long',2160)]:
 vals=[]
 for proc in [1,2]:
  row=next(x for x in json.loads((B/f'TIMING-{proc}.json').read_text())['rows'] if x['case']==case)
  med=row['median_ms'];print(case,proc,med)
  a=med['before'];b=med['specialized'];vals.append((a,b,100*(1-b/a)))
 whole+=f'|{kv}|{vals[0][0]:.6f}→{vals[0][1]:.6f}|{vals[1][0]:.6f}→{vals[1][1]:.6f}|{min(v[2] for v in vals):.2f}%–{max(v[2] for v in vals):.2f}%|\n'
body=f'''# 六小投影固定形状阶段：部分改善，尚未全部对齐

{now}。本轮一个候选已完成并保留。完整单层B14三KV、两个独立进程均减少约1.88%–2.08%，9真实签名有限逐位门通过。Self-Q整层内反而慢约6–7µs，其余五小投影改善；新独立调用体也慢于旧独立实现，不能用“新独立与新整层接近”宣布问题全部解决。

## 小投影和改动

这里的小指GEMM行数M小，不是权重小。B14 hidden为[14,16,3072]，Linear可按[224,3072]行处理。Self Q/K/V分别使用自己的[3072,3072]权重与bias；输出投影及Cross Q/out也是同形状。共六个。Cross K/V来自[14,512,3072]的context，M=7168，不在本次六小投影中。PyTorch权重按[out,in]存储，数学为XWᵀ+b。Q/K后续还有RMS，Self Q/K有RoPE；QKV投影不是整个attention。

仅修改[活动源](../../../../算子研究/wam_ops/optimized/csrc/whole_layer.cu)156–175行：新增`action_projection`固定M224、N/K3072的专用设备阶段，原`linear`命中该形状时调用。固定形状让编译器直接看到64×128 CUTLASS计算体。它是同一个kernel内部的设备函数，没有新增kernel launch。

96个输出tile、两个K1536分段、BF16/bias舍入、四MMA warp、claim队列和grid.sync均未改。108常驻CTA、256线程、98,848B shared、255寄存器不变；localSize1464→1480B。该静态字段不能替代动态spill流量，也不是此次加速的单一解释。跨阶段ready-DAG尚未实现。

## 完整单层主结果

{whole}

这是修改前后相同fixture、同轮交错CUDA Graph的完整一层耗时，两个独立进程，每臂固定600次预热、24次交错采样。边界时钟1410/1215MHz，无GPU0竞争；未设置时钟。不是相对Torch的新整层倍数，不是30层、chunk或机器人E2E。[逐格表](WHOLE-LAYER-COMPARISON.csv)、[原始进程1](TIMING-1.json)、[进程2](TIMING-2.json)。不同阶段历史收益不直接相乘。

## 分项保留退化

{stage_table}

表中均为设备调用点起止，单位µs；范围覆盖两进程×三KV。独立前序使用相同真实native prefix，整层使用自有真实前序；两者缓存/指令状态不必完全相同。旧/新分项来自不同测量轮，只作归因诊断，不能把其差额直接相加当完整层收益。详见[逐格旧新表](STAGE-BEFORE-AFTER.csv)、[当前十GEMM表](stage-control/STAGE-COMPARISON.csv)，旧较快独立结果没有删除。

新独立小投影的外部marker区间相对本轮native慢约33%–45%，比旧独立结果更差；不能以移动参考掩盖退化。Self-Q新整层比新独立调用点仍慢约4.6%–8.6%，Self-out最多约12.2%。其余阶段整层调用点可略快于独立，因此不能把“嵌入差”全称纯同步等待。

当前Cross K/V调用点差约−1.24%至0%，FFN1/2约−1.38%至+0.72%；FFN1独立仍比本轮Torch慢34.05%–35.19%，FFN2慢24.02%–27.68%。这些原语差距未由本次改动解决。

## 能确认什么、不能确认什么

- OBSERVED：只改变固定形状设备函数组织，保留相同数学/tile/sync，完整层稳定改善约2%。支持“编译组织对嵌入成本有影响”，不足以证明全部额外成本来自通用分支或spill。
- OBSERVED：[同DSO静态SASS](SASS-STATIC-SUMMARY.json)显示新增专用设备函数；context/FFN/attention相关函数的静态计数未变。generic GEMM含多种路径，不能将其静态指令数与单一专用函数相除估计速度。静态LDL/STL也不能直接换算耗时。
- HYPOTHESIS：Self-Q退化及新独立版本变慢可能来自编译器跨函数优化/寄存器保存与首阶段缓存状态的组合；尚无当前动态NCU证据分账，不能确认为根因。
- UNKNOWN：动态local访问耗时、等待与真实计算分别贡献多少；未测新版本完整模型/闭环质量。
- RECOMMENDATION：保留有完整层净收益的候选，下一步只针对Self-Q比较原通用路径与专用路径，并保留旧快速独立参考；随后处理FFN原语，最后跨阶段DAG。下一项实际作业仍需有界输入/预算，不自动开始结构搜索。

## 数值、计时及资源边界

[有限门](SIMPLE-GATE-RESULT.json)：B2 action/B14 action/B2 video×三KV，9输出、每例23共有中间量、18有效KV状态转移，以及输入/RNG/重复输出均逐位，max_abs0。72完整调用。阶段诊断240次输出比较逐位，6组有效状态/输入/RNG检查通过。有限真实输入不证明任意输入逐位，不更改四科学权限。

插桩完整层扰动−0.244%至+0.017%。同前序主表及整层边界均1410MHz；三个hot-only context样本边界1395MHz，原记录与标记保留，不混入主表，无追加挑选重跑。[精度索引](PRECISION-SUMMARY.json)、[诊断分析](stage-control/ANALYSIS.json)。

GPU生命周期{gpu:.6f}秒，CPU准备/编译{cpu:.6f}秒；M39累计{res['cumulative_GPU_seconds']:.6f}/36000秒。CUDA active未知。无失败/timeout；PID+出生身份核对所有owned退出，两卡观察时空闲。新22:23:55截止与旧截止均保留。保留两项编译中间产物及原负结果，不继续第二候选。[资源与停止](RESOURCE-AND-STOP-REPORT.json)、[清理核对](CLEANUP-CHECK.json)。

源码e91e09eb、选中DSO23238b37，完整身份见[SOURCE](SOURCE-MANIFEST.json)、[选择](SELECTION.json)、[本地同二进制安装](LOCAL-INSTALL.json)。只改optimized；baseline、原tile、投机生产、保护文件不动。旧源/二进制证据可回溯。
'''
(B/'RESULT.md').write_text(body)
shutil.copyfile('/tmp/analyze_small_projection.py',B/'stage-control/analyze.py')
shutil.copyfile(__file__,B/'report.py')
print('report',gpu,cpu,res['cumulative_GPU_seconds'])
