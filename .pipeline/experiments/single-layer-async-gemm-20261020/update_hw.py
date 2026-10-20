"""Close existing M39 records; no new experiment or GPU registration."""
from pathlib import Path
from datetime import datetime
import re
P=Path(__file__).resolve().parent;R=P.parents[2];C=R/'.pipeline/cycles/wam-research-20261016';now=datetime.now().astimezone().isoformat()
for name in ('PLAN.md','PROGRESS.md'):
 p=C/name;s=p.read_text();s=re.sub(r"updated: '[^']+'",f"updated: '{now}'",s);s=re.sub(r'更新：[^。]+。',f'更新：{now}。',s,1)
 if name=='PLAN.md':
  s=s.replace('  - id: M39\n    status: in_progress','  - id: M39\n    status: completed')
  s=re.sub(r'\| `M39` \|[^\n]+', '| `M39` | completed | [双缓冲结果](../../experiments/single-layer-async-gemm-20261020/RESULT.md)：9点/18状态、432样本，相对旧原型下降17.9%–36.3%，原Torch门仍FAIL；资源登记240/900不一致FAILED，已清理 |',s)
  s=s.replace('用户最新优先M38，先解释并量化当前慢点，不等待数值修复。所有旧终态与门不变；不做新机器人/训练/TEST或30层。','用户最新要求直接在optimized继续，不为带读另建代码版本/实验登记。M39计算已收口，后续在现有记录保留必要证据；更大tile尚未执行。所有旧终态与门不变，不做新机器人/训练/TEST或30层。')
 else:
  s=s.replace('current: M39\nnext: M39','current: M39\nnext: null')
  a=s.index('## M39 —');b=s.index('## 下一步及保护',a)
  s=s[:a]+'''## M39 — 单层GEMM异步tile流水：completed，原数值NO-GO

[结果](../../experiments/single-layer-async-gemm-20261020/RESULT.md)：9真实签名、18合法KV转移相对失败whole有限一致；原Torch/native9通过、新async9逐位FAIL。两个独立进程432完整block计时，新双缓冲比本轮旧whole耗时下降17.9%–36.3%，但仍为native的3.33–7.84倍；同LD40同步对照加速1.14–1.51×。全部18新trace各1kernel，未测chunk/E2E。

插桩GEMM阶段快1.73–1.95×；B14短cross K/V仍52.4%，长self-attention约35%–49%。172 registers、59616B shared、理论1 CTA/SM不变，硬件stall/流量UNKNOWN。675block、264.838秒生命周期、10.68MiB，六owned进程确认退出、两卡空。[资源审计FAILED](../../experiments/single-layer-async-gemm-20261020/RESOURCE-AND-STOP-REPORT.md)：PLAN/runner900但registration240，实际超机器登记435次；原登记不改，不能写预算全通过。可选CPU反汇编因cuobjdump缺失失败也保留。

用户随后明确：直接修改optimized，不因另一agent带读保留独立版本，也不再另建登记。沿现有记录继续；下一候选K32→64及attention数据通路尚未实现/测量，不能称有收益。

'''+s[b:]
  s=s.replace('M34/M35/M37有界作业已终态；M37数值门失败，后继整层异步集成未触发，不重复旧网格搜索；','M34–M39已终态；M37失败CUTLASS未集成，M39是保持旧WMMA数学的双缓冲，不等于M37数值修复；')
 p.write_text(s)
(C/'EXECUTION.md').write_text(f'''# 当前Execution checkpoint

## {now}：M39双缓冲实现与有限验证完成

只改optimized新增GEMM数据通路，SM80 cp.async双buffer、固定64×64×32、同步LD40对照，原attention与stage barrier保持。[结果](../../experiments/single-layer-async-gemm-20261020/RESULT.md)记录两进程432样本/9签名/18写转移/18单kernel trace。对旧whole耗时下降17.9%–36.3%；仍比native慢3.33–7.84倍，原Torch9逐位FAIL；有限匹配失败whole不等于准入。

264.838秒生命周期，675block，五子进程exit0和父进程PID/birth复核均已退出，两卡空。归档发现PLAN/runner900与机器registration240不一致，记资源审计FAILED，不修改原字段；CPU可选反汇编工具缺失失败保留。见[资源报告](../../experiments/single-layer-async-gemm-20261020/RESOURCE-AND-STOP-REPORT.md)。没有后续GPU队列，四false保持。

用户最新要求直接在optimized迭代、不为另一agent带读保留额外版本、不另建登记，已用Maintain存约束记忆；不新建M40。基线、投机与原tile不动。关于更大tile：当前GEMM64×64×32、attention16×64，下一建议先评估K64；这是建议，尚未执行。

本轮起点及旧M38原文见[归档](history/M39-start-20261020/EXECUTION.md)。
''')
p=C/'DISCUSSION-SUMMARY.md';s=p.read_text();s=s.replace('仅optimized新入口','仅修改optimized');s=s[:s.index('## 当前优先级')]+'''## 当前优先级

用户先要求profile整层慢点，再明确继续实现；M38/M39已收口，GEMM双缓冲带来相对旧原型有限提速，仍未匹配Torch数值/速度。资源登记字段不一致作为失败保留，不能以脚本exit0替代全部审计通过。

最新指令：不考虑另一agent带读版本，直接改optimized，不再另起实验登记；这是对原“独立入口保护带读”的替代。Maintain记录在[直接在optimized迭代](../../memory/global/requirements/decision-直接在optimized迭代且不另建登记.md)。baseline/投机与既有准入门继续保护，已有本轮记录继续保留最小必要证据。

用户询问更大tile，当前GEMM64×64×32、attention Q16/KV64。建议优先K32→64，再按形状评估N128；B2 M32不能盲目扩大M，B14不能变成跨候选互相注意。尚未运行这些候选，不把提议记为实测收益或已启动工作。
''';p.write_text(s)
p=R/'.pipeline/PROGRESS.compact.md';s=p.read_text();s=re.sub(r'更新：[^。]+。',f'更新：{now}。',s,1)
a=s.index('M34已完成');b=s.index('## 资源终态',a)
s=s[:a]+'''当前[M39双缓冲结果](experiments/single-layer-async-gemm-20261020/RESULT.md)：仍一层/单persistent kernel，GEMM64×64×32采用LD40与cp.async双buffer。9真实签名/18合法KV转移相对失败whole有限一致；原Torch/native9逐位通过、新async9逐位FAIL。两进程432完整block样本，对本轮旧whole耗时下降17.9%–36.3%，仍比native慢3.33–7.84倍；同步LD40对照区分布局收益。插桩GEMM快1.73–1.95×，长KV self-attention现占35%–49%。[源码](../算子研究/wam_ops/optimized/csrc/gemm_pipeline.cuh)、[说明](../算子研究/wam_ops/optimized/PIPELINE.md)。

M39实际675block/264.838秒/约10.68MiB，六owned进程退出、两卡空，无GPU队列。**资源审计FAILED**：PLAN/runner900，机器registration240，实际超后者435；不改原登记。可选CPU反汇编缺cuobjdump失败保留。NCU硬件计数/SASS数量UNKNOWN；数值和全部四false不变。[清理与止损](experiments/single-layer-async-gemm-20261020/RESOURCE-AND-STOP-REPORT.md)。

旧[M34](experiments/single-layer-warp-specialized-20261019/RESULT.md)原型9点逐位FAIL/18写9FAIL，5.12–11.28倍慢、600/600调用后清理；[精度定位](experiments/single-layer-warp-specialized-20261019/PRECISION-CUTS.md)Q/FFN2/video attention差异保留。旧编译/依赖/CPU失败、远端未部署保护项false均不抹除。M39只匹配失败原型，不修复旧数值或继承其他版本证书。

'''+s[b:]
a=s.index('用户最新优先当前整层慢点');b=s.index('M36[',a)
s=s[:a]+'''M38[profile](experiments/whole-layer-bottleneck-20261020/RESULT.md)已完成：6签名/222block/300原语，GEMM主导、静态领取更慢。M39随后实现双缓冲并收口。用户最新要求[直接改optimized、不另建登记](memory/global/requirements/decision-直接在optimized迭代且不另建登记.md)，无需为另一agent带读冻结旧源码。沿现有记录继续，下一建议评估K64和attention数据通路，尚未执行，不能称已获收益。

'''+s[b:]
p.write_text(s)
p=R/'.pipeline/INDEX.md';s=p.read_text();s=re.sub(r'更新时间：[^。]+。',f'更新时间：{now}。',s,1);anchor='最新[M38整层慢点实测]';a=s.index(anchor);b=s.index('\n\n',a)
s=s[:a]+'''最新[M39双缓冲结果](experiments/single-layer-async-gemm-20261020/RESULT.md)：对旧原型耗时下降17.9%–36.3%，仍比native慢3.33–7.84倍且原Torch逐位FAIL。9点/18状态/432样本，owned已清理；资源登记240/900不一致FAILED保留。按最新指令直接在optimized迭代、不为带读另建版本/登记。[M38慢点依据](experiments/whole-layer-bottleneck-20261020/RESULT.md)保留。'''+s[b:];p.write_text(s)
p=R/'.pipeline/experiments/INDEX.md';s=p.read_text();s=s.replace('# 当前研究实验\n','# 当前研究实验\n\n- [M39单层双缓冲](single-layer-async-gemm-20261020/RESULT.md)：9点/18状态、两进程432样本，相对旧原型下降17.9%–36.3%，原Torch门仍FAIL；资源字段240/900不一致保留，已清理。后续沿已有记录直接改optimized，不另建登记。\n',1);p.write_text(s)
p=R/'算子研究/wam_ops/optimized/PIPELINE.md';s=p.read_text();s=s.replace('仍然是阶段内流水，尚无跨stage ready队列、跨CTA普通shared memory或candidate query合并。额外buffer仍小于attention原占用，因此理论动态shared总量可能不变；寄存器和实际驻留须读取编译/运行结果。','仍然是阶段内流水，尚无跨stage ready队列、跨CTA普通shared memory或candidate query合并。实测async172 registers/thread、59616B shared、理论1 CTA/SM；profile180 registers。理论驻留不是achieved occupancy。')
s+='''
## 本轮结果与后续切块

[完整证据](../../../.pipeline/experiments/single-layer-async-gemm-20261020/RESULT.md)：两独立进程9点432样本，对旧原型耗时下降17.9%–36.3%，仍比native慢3.33–7.84倍。9点/18合法KV转移匹配旧失败原型；原Torch逐位9/9失败，没有生产准入。登记上限240/900不一致作为资源审计FAILED保留，GPU已清理。

当前GEMM64×64×32、每计算warp32×32，WMMA16×16×16；attention Q16/KV64/head128；Norm完整行3072；pointwise128元素。可先评估GEMM K64以减少循环/同步，再按M/N实际大小考虑更宽N；B2 action M32扩大M会浪费。长KV attention的加载/softmax仍是独立瓶颈，不能直接把不同candidate拼成更长query序列。这些更大tile尚未实现或测量。

用户最新要求后续直接改optimized，不再为另一agent带读保留独立版本或新建实验登记；此前对照源码身份仍可通过实验manifest和Git追溯。
''';p.write_text(s)
# Existing report typo only; preserve all raw records.
p=P/'RESULT.md';p.write_text(p.read_text().replace('K14336898→451','K14336的898→451'))
plan=(C/'PLAN.md').read_text();progress=(C/'PROGRESS.md').read_text()
ids=re.findall(r'^  - id: (M\d+)$',plan,re.M);found=re.findall(r'^## (M\d+) —',progress,re.M)
assert ids==found,(ids,found)
print('Plan/Progress IDs identical',ids)
print('compact bytes',len((R/'.pipeline/PROGRESS.compact.md').read_bytes()))
print('Cycle four bytes',sum((C/n).stat().st_size for n in ('PLAN.md','PROGRESS.md','EXECUTION.md','DISCUSSION-SUMMARY.md')))
