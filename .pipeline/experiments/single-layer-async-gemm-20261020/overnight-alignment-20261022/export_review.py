"""Explicit public allowlist; prepares mirror, never commits or pushes itself."""
from pathlib import Path
import datetime
import hashlib
import json
import re
import subprocess

B = Path(__file__).resolve().parent
R = B.parents[3]
M = R/'review_exports/WAM-egale-libero-fullscale-20261017'
C = R/'.pipeline/cycles/wam-research-20261016'
robo = R/'.pipeline/experiments/robotwin-pilot-20261020/runtime-opt-cross-task-20261021'
previous = subprocess.check_output(['git','rev-parse','HEAD'],cwd=M,text=True).strip()
selected = json.loads((B/'SELECTED.json').read_text())
resource = json.loads((B/'RESOURCE-AND-STOP-REPORT.json').read_text())
now = datetime.datetime.now().astimezone().isoformat()
files = [R/'.pipeline'/x for x in ('INDEX.md','PROGRESS.compact.md','memory/INDEX.md')]
files += list(C.glob('*.md')) + list((C/'history/overnight-before-20261022').glob('*.md'))
files += [R/'.pipeline/memory/global/requirements/decision-算子夜间持续对齐至十月二十二日九点.md']
files += [R/'算子研究/wam_ops/optimized'/x for x in selected['sources']]
files += [R/'算子研究/wam_ops/optimized'/x for x in ('whole_layer.py','WHOLE_LAYER.md','READ_CODE_PROMPT.md')]
omissions = []
excluded = {'build','cache','tmp','generated','simple-stage-build','__pycache__','before'}
for base in (B,robo):
    for p in base.rglob('*'):
        if not p.is_file():
            continue
        if set(p.relative_to(base).parts)&excluded:
            continue
        if p.suffix not in ('.md','.json','.csv','.py','.patch','.cu','.cuh','.jsonl') or p.stat().st_size>8*1024**2:
            omissions.append(dict(path=str(p.relative_to(R)),bytes=p.stat().st_size,reason='binary/log/large artifact stays on server'))
            continue
        # Persisted per-observation streams remain server evidence; compact summaries exported.
        if p.suffix=='.jsonl' and p.name!='CANDIDATE-SUMMARY.jsonl':
            omissions.append(dict(path=str(p.relative_to(R)),bytes=p.stat().st_size,reason='raw event stream stays on server'))
            continue
        files.append(p)
manifest=[]
for p in sorted(set(files)):
    raw=p.read_bytes()
    # Credential/private-key markers are disallowed in this explicit export set.
    assert not re.search(rb'(?m)^-----BEGIN (?:OPENSSH |RSA |EC |ENCRYPTED )?PRIVATE KEY-----',raw), p
    exported=raw
    if p.suffix in ('.md','.json','.csv'):
        exported=('\n'.join(x.rstrip() for x in raw.decode().splitlines()).rstrip()+'\n').encode()
    q=M/p.relative_to(R);q.parent.mkdir(parents=True,exist_ok=True);q.write_bytes(exported)
    manifest.append(dict(path=str(p.relative_to(R)),bytes=len(exported),sha256=hashlib.sha256(exported).hexdigest(),
                         local_original_sha256=hashlib.sha256(raw).hexdigest(),
                         transform='LF and trailing whitespace only; server original preserved' if raw!=exported else None))
state=dict(observed_at=now,previous_derived_snapshot_commit=previous,operator=selected,
    operator_result=str((B/'RESULT.md').relative_to(R)),resource=resource,
    operator_whole_ms={'short':2.2958080768585205,'middle':2.453248083591461,'long':2.4762880206108093},
    uncached_guarded_relative_latency_percent={'short':10.935183106149182,'middle':12.173719326097765,'long':12.502919170868326},
    full_alignment_achieved=False,ready_DAG_implemented=False,
    robotwin=dict(cross_task_complete_episodes=4,independent_quality_samples_added=0,
                 cumulative_GPU_lifecycle_seconds=54718.243980407715,owned_jobs_running=False,
                 original_deadline='2026-10-22T01:26:04.706686+08:00',deadline_expired=True,
                 expanded1800_started=False,additional160hours_authorized=False),
    engineering_quality_screen=False,independent_test_complete=False,
    noninferiority_established=False,formal_performance_permission=False,
    historical_paused72='review/RESULTS.json unchanged')
(M/'review/CURRENT-EXECUTION-20261022.json').write_text(json.dumps(state,ensure_ascii=False,indent=2)+'\n')
link='../'+str(B.relative_to(R));rblink='../'+str(robo.relative_to(R))
(M/'review/CURRENT-EXECUTION-20261022.md').write_text(f'''# 当前执行证据（{now}）

**本轮单层有改善，但尚未完全对齐最强B14同工作量基线。** 最终源0cc1944b、DSOac3f9e46已在两台机器optimized安装。原paused72及四项false保留；安装研究版本不等于生产资格。

|B14 KV|本夜开始ms|最终ms|uncached guarded强基线ms|最终仍慢|
|---|---:|---:|---:|---:|
|288|2.510|2.296|2.070|10.94%|
|1152|2.718|2.453|2.187|12.17%|
|2160|2.977|2.476|2.201|12.50%|

同卡/同输入、两独立进程、轮换Graph计时；相对本夜开始延迟下降8.52%/9.75%/16.81%。这是单层，不是30层/机器人E2E；不使用profiler时间。cached projection未纳入本轮矩阵。

- [完整结果]({link}/RESULT.md)、[逐进程完整层表]({link}/WHOLE-LAYER-COMPARISON.csv)、[三KV逐阶段]({link}/stage-control-final-r2/RESULT.md)、[claim证据矩阵]({link}/CLAIM-EVIDENCE-MATRIX.md)。
- 9真实签名＋21结构变体有限逐位；原Torch最终输出、23共有中间量、普通/debug、有效KV/输入/RNG通过。08非16尾曾失败，08b修复，原FAILED保留。不是全部输入定理，也没有在线质量放行。
- 改动：静态point/norm分派、padded写回、既有guarded self嵌入、双4warp attention、FFN1 N256K32半区epilogue、去冗余入口屏障。仍108CTA单层，必要出口依赖保留；没有ready-DAG。
- 剩余慢项self-attention/Cross-K RMS/FFN/GELU，独立执行也慢；资源干预说明整层统一资源配置是部分原因。不能全归因grid等待，不能把静态local936B当动态spill字节。
- [资源与终态]({link}/RESOURCE-AND-STOP-REPORT.md)：{resource['status']}，当前累计保守计费{resource['cumulative_charged_seconds']:.3f}/36000秒，含本机等待。09:00原硬截止；NCU若未取得则UNKNOWN。
- [RoboTwin工程4局]({rblink}/RESULT.md)终态/清理，basket双方成功−6.19%、microwave双方失败−3.74%，完整受用摘要一致；固定顺序单次探索性，非独立质量N。原01:26截止已过，1800局/160GPU小时未启动/未批准。

[源码说明](../算子研究/wam_ops/optimized/WHOLE_LAYER.md)、[CUDA单层](../算子研究/wam_ops/optimized/csrc/whole_layer.cu)、[带读prompt](../算子研究/wam_ops/optimized/READ_CODE_PROMPT.md)。原始tensor/DSO/详细流留服务器；公开包边界见EXPORT-NOTES-20261022-OVERNIGHT.json。前派生快照commit `{previous}`。
''')
start=M/'review/START_HERE.md';old=start.read_text();separator=old.index('\n---\n')
start.write_text('''# 当前入口

最新：[夜间单层对齐与RoboTwin收口](CURRENT-EXECUTION-20261022.md)、[状态JSON](CURRENT-EXECUTION-20261022.json)、[HW短入口](../.pipeline/PROGRESS.compact.md)、[算子源码说明](../算子研究/wam_ops/optimized/WHOLE_LAYER.md)。

最终B14三KV约2.296/2.453/2.476ms，比夜间开始下降8.5%/9.7%/16.8%，仍比同工作量uncached guarded慢约11%–12.6%。9签名＋21结构变体有限逐位、两机研究版同DSO；ready-DAG未实现，四权限false。NCU/资源状态以新JSON为准。RoboTwin4局工程验证已终态，原截止已过，1800局未启动。下面旧TF交接保持历史身份。
'''+old[separator:])
notes=dict(previous_commit=previous,observed_at=now,source_manifest=manifest,omissions=omissions,
    always_omitted=['model weights','input tensor fixtures','DSOs/build products','raw command/query streams','credentials'],
    scope='Source plus derived/relevant raw small evidence; not an independently replayable model distribution',
    historical_review_RESULTS_preserved=True,protected_state_cycle_rules_preserved=True)
(M/'review/EXPORT-NOTES-20261022-OVERNIGHT.json').write_text(json.dumps(notes,ensure_ascii=False,indent=2)+'\n')
changed=[x['path'] for x in manifest]+['review/CURRENT-EXECUTION-20261022.json','review/CURRENT-EXECUTION-20261022.md','review/START_HERE.md','review/EXPORT-NOTES-20261022-OVERNIGHT.json']
(B/'PUBLIC-ALLOWLIST.json').write_text(json.dumps(sorted(set(changed)),ensure_ascii=False,indent=2)+'\n')
print(json.dumps(dict(files=len(changed),bytes=sum(x['bytes'] for x in manifest),previous=previous,omitted=len(omissions)),ensure_ascii=False))
