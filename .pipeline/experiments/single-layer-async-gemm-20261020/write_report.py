"""Render this run's evidence, including inconsistent budget metadata; no GPU work."""
from pathlib import Path
import csv,json
from datetime import datetime
P=Path(__file__).resolve().parent
r=json.loads((P/'RESULT.json').read_text());state=r['resource'];reg=json.loads((P/'remote-evidence/REGISTRATION.json').read_text())
rows=r['rows'];ss=list(csv.DictReader((P/'STAGES.csv').open()))
case_names={'short':'短','middle':'中','long':'长'}
def label(c):
 kv,b,kind=c.split('-');return f'{b} {kind} / {case_names[kv]}'
table='\n'.join(f"| {label(x['case'])} | {x['native']:.3f} | {x['whole']:.3f} | {x['sync']:.3f} | {x['async']:.3f} | {100*x['async_latency_reduction']:.1f}% | {x['async_over_native']:.2f}× |" for x in rows)
process='\n'.join(f"| {label(x['case'])} | {x['by_process'][0]['whole_to_async_speedup']:.3f}× | {x['by_process'][1]['whole_to_async_speedup']:.3f}× | {x['sync_to_async']:.3f}× |" for x in rows)
stagetable=[]
for x in rows:
 values={s['stage']:float(s['async_ms']) for s in ss if s['case']==x['case']};kt=x['profile_kernel_ms']['profile']
 stagetable.append(f"| {label(x['case'])} | {100*x['profile_GEMM_ms']['profile']/kt:.1f}% | {100*values['self_attention']/kt:.1f}% | {100*(values['cross_K_proj']+values['cross_V_proj'])/kt:.1f}% | {x['profile_GEMM_speedup']:.3f}× |")
(P/'RESULT.md').write_text(f'''# M39：单层tile GEMM双缓冲实测

更新时间：{datetime.now().astimezone().isoformat()}。计算终态 `{r['status']}`；原Torch精度和生产准入仍为NO-GO。**另有资源登记字段不一致，资源审计FAILED**，见[资源报告](RESOURCE-AND-STOP-REPORT.md)，不改原始登记或脚本终态。

## OBSERVED：实现与验证

在optimized中实现SM80 cp.async双缓冲，仍为完整一层、一个cooperative persistent kernel。warp0领取CTA任务、1/2加载A/W、3–6执行WMMA、7写回；64×64×32输出/归约分块及K递增累加、BF16 epilogue保持。双buffer与LD40共享布局使下一K块搬运可与当前MMA重叠；未改attention、norm、RoPE、cache、scheduler或stage barrier。

[算法](../../../算子研究/wam_ops/optimized/csrc/gemm_pipeline.cuh)、[封装](../../../算子研究/wam_ops/optimized/whole_layer_pipeline.py)、[构建](../../../算子研究/wam_ops/optimized/build_pipeline.py)、[源manifest](SOURCE-MANIFEST.json)。sync为同LD40同步对照，async为新双缓冲，profile只是async的计时副本；不是三种生产候选。用户随后明确后续直接改optimized、不为带读保留额外版本、不另建登记；本轮旧源码身份及测量关系仍以源manifest保存。

共9真实block0：action B2/B14、Q16，短/中/长有效历史288/1152/2160；video B2、Q128，历史272/1136/2144。D3072、H24、head_dim128、FFN14336、cross512。B14仍14个独立Q16注意力实例，不是Q112；私有KV语义保持。

- native对原Torch最终输出9/9逐位通过；新async对原Torch最终输出9/9逐位失败，继承旧原型差异，不能写“精度已对齐”。
- sync/async/profile相对旧whole在9点全部保存中间量、输出、metadata及全部逻辑有效KV相同；无效KV不比较。输入、RNG、参数身份、tile exactly-once及新async重复确定性通过。
- 6个B2 action/video起点，合法update1→2→0，共18个转移，旧whole和新async输出及有效状态相同。
- 两独立计时进程各9个Chrome trace均恰好1个compute kernel，共18份；memcpy/memset/annotation不计作计算kernel。
- 这些只是有限真实fixture验证；不代表所有输入，不包含接受路径、完整30层、chunk或机器人闭环。

## OBSERVED：完整单层时间

远端A100-SXM4-40GB GPU0，无观察到其他GPU context；CPU4/5两核。Torch2.9.0+cu126，扩展CUDA12.4/SM80。两个独立进程，每cell每进程6次交错CUDA-event计时，432个样本全部保留。下表每cell为12样本的合并中位数，单位ms；profiler和状态准备在计时外。相对改进只比较本轮同输入、同一时段旧whole；native为成熟库完整block，与原Torch有限逐位一致。

| 输入 | native | 旧whole | 同布局sync | 新async | 相对旧whole耗时下降 | 新async/native耗时 |
|---|---:|---:|---:|---:|---:|---:|
{table}

新完整block相对旧原型耗时下降17.9%–36.3%，加速1.22–1.57×；与同LD40同步对照比加速1.14–1.51×，说明收益不全来自padding。仍比native慢3.33–7.84倍，不能称为超越Torch的加速。合并中位数的比值不必等于各进程比值的平均；原样本与逐配对比值全部在[RESULT.json](RESULT.json)/[TIMINGS.csv](TIMINGS.csv)，不选最好轮次。

| 输入 | 进程1旧whole/async | 进程2旧whole/async | 合并sync/async |
|---|---:|---:|---:|
{process}

两进程各cell中位数都改善；所有本轮逐repeat旧whole/async比也大于1。没有置信区间或普适性能保证。此处是未准入原型的机制测量，uncached eager对照，不是最强cached/Graph/shared-prefix路径，不是机器人E2E。

## OBSERVED：阶段归因

独立globaltimer插桩副本，每case每路径3次。GEMM十阶段总时间改善1.73–1.95×。下表比例的分母是新插桩副本kernel时间；不可与上方未插桩时间直接相减。

| 输入 | 新GEMM总占比 | 新self-attention占比 | 新cross-K/V投影占比 | GEMM旧/新 |
|---|---:|---:|---:|---:|
{chr(10).join(stagetable)}

新async172 registers/thread、59616B shared、local bytes0，理论1 CTA/SM，108CTA×256threads；sync174、profile180 registers。寄存器与shared并未出现足以解释该提速的驻留变化；理论值不是achieved occupancy。K3072内部CTA barrier194→99、K14336898→451是源码静态计数，并非实测stall比例。

## HYPOTHESIS / UNKNOWN

双缓冲、cp.async缓存提示和较少CTA barrier共同组成这次改动；sync控制了LD40布局，但没有分别隔离这三个因素，不能把所有改进归因于纯加载重叠。NCU硬件计数仍UNKNOWN；L2/DRAM流量、stall和实际occupancy未测。可选SASS核查因现有工具链没有cuobjdump而未执行，不虚报LDGSTS数量；保留CPU失败记录。

新B14短KV仍由cross K/V投影主导（52.4%）；长KV action的self-attention约35%–36%，video长KV49.4%。下一优化应分别处理GEMM效率与attention数据通路；本轮没有证明更大tile或ready队列能加速。

## RECOMMENDATION：直接在optimized继续

先评估GEMM K32→K64，保持输出M64/N64及原16维MMA次序，用更少循环与同步换额外shared；不盲目放大所有维度。B2 action实际M32，扩大M会增加补零；cross-K/V M1024/7168可再考虑N128，但需要检查寄存器和任务并行度。Attention action保持Q16边界，先处理KV加载与softmax；任何改变归约树的做法重新核对数值。

这些是下一步建议，未运行，未另建登记；沿当前optimized与现有记录继续。baseline、投机源、原tile、旧FAILED和四项false保持。结果不能给生产路径或H3/H4签证。
''')
gpu_worker=sum(x['lifecycle_seconds'] for x in state['jobs'] if not x['name'].startswith('build'))
audit=dict(status='FAILED_REGISTRATION_CAP_MISMATCH',written_plan_full_block_cap=900,
 runtime_registration_full_block_cap=reg['full_block_cap'],runner_enforced_full_block_cap=900,
 actual_full_block_calls=state['full_block_calls'],over_runtime_registered_cap=state['full_block_calls']-reg['full_block_cap'],
 lifecycle_seconds=state['lifecycle_seconds'],lifecycle_cap_seconds=reg['lifecycle_cap_seconds'],
 original_registration_preserved=True,no_more_GPU_work=True,
 explanation='run_remote copied M38 full_block_cap=240/primitive_cap=600 metadata; plan and runner enforced 900; this is a real registration inconsistency, not retrospective correction')
(P/'RESOURCE-AUDIT.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n')
jobtable='\n'.join(f"| {x['name']} | {x['lifecycle_seconds']:.3f} | {x['returncode']} | {x['timed_out']} | {x['owned']['pid']} | {x['owned']['birth_ticks']} |" for x in state['jobs'])
(P/'RESOURCE-AND-STOP-REPORT.md').write_text(f'''# M39资源与终态

## OBSERVED

T0北京时间2026-10-20 10:31:32.306794；登记停新11:26:32、清理11:31:32。实际10:35:57完成，生命周期{state['lifecycle_seconds']:.3f}秒，低于3600秒上限。包含CPU构建168.288秒；四GPU worker进程生命周期合计{gpu_worker:.3f}秒。后者仍含初始化/审计，不等于CUDA active时间；CUDA active累计UNKNOWN。

| 子进程 | 生命周期秒 | exit | timeout | PID | birth ticks |
|---|---:|---:|---|---:|---:|
{jobtable}

675次完整block：gate99、timing各261、stages54；432计时样本、18Chrome traces、0机器人/训练/30层。远端新增文件{state['storage_bytes']}字节（约{state['storage_bytes']/2**20:.2f}MiB，STATUS时点），低于512MiB；CPU归档后少量JSON另存。父进程639721/birth706195083及五子进程均按身份复查已退出，GPU上下文空，两卡当时0MiB/0%，未向他人进程发信号。[清理](remote-evidence/POST-CLEANUP.json)、[原始终态](remote-evidence/STATUS.json)。本机保护清单差异为空，远端全部已列保护项不变。

## FAILED / UNKNOWN：预算字段不一致

归档发现本轮PLAN在执行前登记900次完整block，runner也硬编码900；但原始REGISTRATION.json误沿用M38字段full_block_cap=240及primitive_cap=600。实际675超过机器登记240达435次。因此资源审计记`FAILED_REGISTRATION_CAP_MISMATCH`，**不能报告所有登记预算均通过**。保留原registration、源hash和COMPLETE计算终态，另存[审计](RESOURCE-AUDIT.json)，不改成900、不回填旧记录。这是执行登记实现错误，当前已停止GPU工作，没有用剩余书面额度继续实验。

GPU子进程没有timeout/非零退出。后续可选CPU反汇编在归档前因cuobjdump不存在失败，未执行SASS核查，未新增GPU调用；[原失败](remote-evidence/POST-ARCHIVE-ATTEMPT1.json)、[UNKNOWN结论](remote-evidence/INSTRUCTION-AUDIT.json)。随后仅CPU归档成功；不是重跑GPU实验。NCU计数、SASS指令数量、实际CUDA active总和为UNKNOWN。

## STOP / RECOMMENDATION

本轮停止于有限性能改善，原Torch逐位9/9失败，禁止生产集成/正式质量性能放行。四科学权限false，旧失败不修改。GPU生命周期剩余上限约{3600-state['lifecycle_seconds']:.1f}秒只是算术余额，不表示已排队或可重启本作业。所有GPU程序已清理。

用户最新要求后续直接改optimized、不另建登记；用已有记录保留最小必要成本/失败证据，代码与记录中的预算必须同源，避免再次复制旧常量。baseline/投机/原tile保持。更大tile尚未执行，不能称已获收益。
''')
print('reports written; resource audit explicitly FAILED')
