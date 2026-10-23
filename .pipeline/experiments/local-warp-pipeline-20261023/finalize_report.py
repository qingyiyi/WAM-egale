from pathlib import Path
import json,csv,hashlib,datetime
E=Path(__file__).resolve().parent
source_commit=json.loads((E/'SOURCE-COMMIT.json').read_text())['commit']
summary=json.loads((E/'T4-final-SUMMARY.json').read_text())
cleanup=json.loads((E/'OWNED-CLEANUP.json').read_text())
l=json.loads((E/'RESOURCE-LEDGER.json').read_text());wall=cleanup['wall_elapsed_seconds']
l.update(status='CLOSED',closed_at=cleanup['time'],GPU_reserved_window_seconds=wall,non_worker_window_seconds=wall-l['GPU_lifecycle_seconds']-l['CPU_compile_seconds'],final_cumulative_charged_seconds=l['prior_cumulative_seconds']+max(wall,l['GPU_lifecycle_seconds']+l['CPU_compile_seconds']),all_owned_exited=cleanup['all_owned_exited'],gpu0_empty_at_close=cleanup['gpu0_empty'])
l['remaining_original_seconds']=l['cumulative_cap_seconds']-l['final_cumulative_charged_seconds'];l['remaining_this_window_seconds']=l['gpu_cap_seconds']-wall
l['accounting_note']='保守将从开始至GPU清理整个占用窗口计入本轮120分钟与原10小时累计；worker生命周期/CPU编译另列，不能相加重复收费。非worker余量含准备、交互、文献、文件传输，未分别测CPU active/排队。CUDA active UNKNOWN。原逐job累计字段保留。'
(E/'RESOURCE-LEDGER.json').write_text(json.dumps(l,indent=2)+'\n')
manifest=json.loads((E/'SOURCE-MANIFEST.json').read_text())
for name in ['L1','L13']:
 p=E/f'variants/{name}/算子研究/wam_ops/optimized/csrc/whole_layer.cu';manifest['variants'][name]=dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
manifest.update(retained='L13',retained_source_commit=source_commit,retained_binary=cleanup['binaries']['L13'],production_source_and_binary_unchanged=True,worktree_now_contains='L13 source commit; original anchor A remains in Git and remote independent A binary',timing_input_identity='T0-process1/2 and T4-final1/2 TIMING files reference identical actual B14 layer0 fixture paths; model loaded with same block_loader; no whole-model hash was run',build=dict(nvcc='12.4.99',target='SM80',python='3.12.14',torch='2.9.0+cu126',driver='535.161.08',note='Existing toolchain mismatch warning retained, no installation or upgrade'))
manifest['normal_binaries']={}
for fn in ['T0-process1/TIMING-1.json','T2-timing1/TIMING-1.json','T3-L13-timing1/TIMING-1.json']:
 p=E/fn
 if p.exists():manifest['normal_binaries'].update(json.loads(p.read_text()).get('binary_identity',{}))
(E/'SOURCE-MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
selected=dict(status='RETAINED_ISOLATED_RESEARCH_NOT_PRODUCTION',candidate='L13',source_commit=source_commit,source_sha256=manifest['variants']['L13']['sha256'],binary=cleanup['binaries']['L13'],mainline_FFN2='INCONCLUSIVE_NET_GAIN_NOT_ADOPTED',LN1='FINITE_PASS_SMALL_NET_GAIN',LN3='FINITE_PASS_SMALL_ADDITIONAL_GAIN_NO_CREDIBLE_REGRESSION',engineering_quality_screen=False,independent_test_complete=False,noninferiority_established=False,formal_performance_permission=False,production_admitted=False,scope='B14 warm complete single block on A100; finite correctness covers B2 action/video and B14 action, not complete WAM or robot tasks')
(E/'SELECTED.json').write_text(json.dumps(selected,indent=2)+'\n')
rows=[]
for kv,r in zip([288,1152,2160],summary):
 m=r['median_ms'];v=m['L13'];rows.append(f'|{kv}|{m["A"]:.6f}|{m["L1"]:.6f}|{v:.6f}|{m["D"]:.6f}|{m["G"]:.6f}|{m["A"]/v:.4f}|{m["D"]/v:.4f}|{m["G"]/v:.4f}|')
table='|KV|A ms|L1 ms|L13 ms|D ms|G ms|A/L13|D/L13|G/L13|\n|---|---:|---:|---:|---:|---:|---:|---:|---:|\n'+'\n'.join(rows)
(E/'FINAL-COMPARISON.md').write_text('# 最终同轮完整层比较\n\n'+table+'\n\nT4-final1/2：两独立进程，各档各臂150次预热、24轮交错样本×10次Graph replay。表为两进程中位数均值；每进程与IQR/配对变化见T4-final-SUMMARY.json，逐样本见原TIMING JSON。独占GPU0；GPU1仍有RoboTwin，不把整台机器称独占。GPU事件边界为warm完整block CUDA Graph，分配/首次缓存构建不在warm延迟，原始TIMING另存cold构建和展开成本。双方同Cross最终K/RMS/V预计算、相同真实B14/Q16、相同有效历史与私有suffix。\n\nA固定319ac6a独立persistent；D原Torch静态dense-history；G现有可用guarded执行路径，两参考均经输出/有效状态核验。L13只采用LN1/LN3的调制融合，无BC主线组合。未测跨硬件、30层、完整chunk或E2E。\n\n相对A三档延迟降0.687%/0.572%/0.428%，与前轮方向一致，幅度小，不声称统计显著。G/L13中长小于1：L13仍慢G3.602%/3.391%。只报D的1.30–3.00倍会掩盖强基线差距。\n')
with (E/'WHOLE-LAYER-SAMPLES.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=['process','case','repeat','arm','ms']);w.writeheader()
 for i in [1,2]:
  d=json.loads((E/f'T4-final{i}/TIMING-{i}.json').read_text())
  for r in d['rows']:
   for s in r['samples']:w.writerow(dict(process=i,case=r['case'],**s))
resource=f'''# 本轮资源与停止报告

终态：CLOSED，{cleanup['time']}。26个登记job；全部owned子进程及guard按PID/出生身份核验已退出，GPU0无compute进程、显存11MiB；GPU1原机器人队列不修改。

- 本轮保守占用窗口：{wall:.3f}秒 = {wall/60:.3f}分钟，低于120分钟。
- GPU worker生命周期：{l['GPU_lifecycle_seconds']:.3f}秒 = {l['GPU_lifecycle_seconds']/60:.3f}分钟，包含失败/加载/退出；不是CUDA active。
- CPU编译job：{l['CPU_compile_seconds']:.3f}秒 = {l['CPU_compile_seconds']/60:.3f}分钟，包含首失败。
- 其余窗口墙钟：{l['non_worker_window_seconds']:.3f}秒，含准备/传输/用户交互/文献；没有分项active测量。CUDA active = UNKNOWN。不能把这项伪称GPU忙碌。
- 原累计{l['prior_cumulative_seconds']:.3f}秒；本轮保守闭合后{l['final_cumulative_charged_seconds']:.3f}/36000秒，剩余{l['remaining_original_seconds']/3600:.3f}小时。旧起点/截止不重置；本轮提前收口，不消费剩余额度。
- 远端新增研究目录{cleanup['storage_bytes']/2**20:.3f}MiB，低于3GiB；无新增模型/episode大文件。
- 失败：T2-BC-build临时目录相对路径失败，1.704573秒CPU、0GPU；T2-diagnostic对未初始化allocated尾部误比较失败，11.643361秒GPU生命周期。两原记录保留，新attempt修runner/有效范围，不放宽数学门。
- 全部job无timeout；没有重复机器人任务、训练、NCU提权、驱动修改、FA2重写或新DAG。

原逐job状态及真实returncode见RESOURCE-LEDGER.json；清理见OWNED-CLEANUP.json。没有把退出脚本写成租卡关机，本轮没有管理租卡实例。
'''
(E/'RESOURCE-AND-STOP-REPORT.md').write_text(resource)
report=f'''# 局部warp流水判定回传

完成时间：{cleanup['time']}。锚点319ac6a17daed34e80692796f72a97bf44f4a99f；文档锚点effd6580b43b503ce6f00b0074f8cc8570adffa1。

**结论：FFN2主线未得到可信完整层净收益，不采用；副线保留LN1＋LN3调制融合L13，完整层比A快约0.4%–0.7%。中长KV仍未超过已有强guarded基线。** T0至T4全部完成；没有扩展DAG/FA2/tile搜索。仅隔离研究采用，原baseline、投机生产与两机原使用入口不改，四项科学权限及production_admitted全部false。

## 固定源码和结果入口

- 保留源码commit：`{source_commit}`，文件`算子研究/wam_ops/optimized/csrc/whole_layer.cu`。本地独立worktree已提交；GitHub发布状态另见PUBLICATION-RECEIPT.json，未有收据前不表示远端可读。
- 完整源与DSO身份：[SOURCE-MANIFEST.json](SOURCE-MANIFEST.json)，A独立原二进制保留；L13源码SHA256 `{selected['source_sha256']}`，DSO SHA256 `{selected['binary']['sha256']}`。
- 本轮唯一新采用patch：[T3-L13-build/CHANGE.patch](T3-L13-build/CHANGE.patch)；负候选由make_bc.py和锚点可重建，BC普通/诊断源码均保存于variants与独立补丁。
- 完整阶段表：[STAGE-TABLE.md](STAGE-TABLE.md)；结果：[FINAL-COMPARISON.md](FINAL-COMPARISON.md)；逐样本：[WHOLE-LAYER-SAMPLES.csv](WHOLE-LAYER-SAMPLES.csv)。本报告末尾附全表。

## OBSERVED：主线A/B/C

A为独立锚点二进制；B/C同一新DSO，W3–6四warp MMA、128×128tile、K7168×2、BF16 partial、任务队列和八warp提交保留。W0/1/2/7提前准备第二段当前tile的bias/table及首8行gate/R2，7KiB放在原shared union允许空间。只覆盖满tile6.25%的行，尾tile8.33%；没有提前读取尚未发布partial。

B先准备并全CTA会合再MMA；C准备与MMA重叠再会合。**控制不完全：B的额外串行barrier也出现在无准备的第一段**，C/B收益包含这次同步消失，不是干净的重叠因果分解。A未被塞入新分支大kernel。

每臂9真实/36结构，225次完整调用有限逐位、全部有效KV、输入/RNG与合法状态通过。诊断补测42调用验证A诊断副本与A有效中间量，以及B/C首段BF16 partial、最终FFN2/output；没有把allocated未初始化尾部当有效状态。

每档288个第二段任务的软件代表warp区间：B重叠0，C重叠288，C重叠P50约3.072–4.096μs。诊断BC local1048B、普通1016B，不能把诊断171μs MMA时间当普通路径时间/NCU吞吐。只支持代表warp区间重叠，不支持全CTA关键路径或TensorCore活跃度上升。

无插桩C/A：1.000821/0.999460/1.001044，延迟变化+0.082%/−0.054%/+0.104%，小于/接近T0 A/A波动（观察范围约−0.290%至+0.102%）。三档均INCONCLUSIVE净收益，**C不采用**；未消耗额外算法修订或展开扫参。见[T2-RESULT.md](T2-RESULT.md)、[T2-DIAGNOSTIC-SUMMARY.json](T2-DIAGNOSTIC-SUMMARY.json)。

## OBSERVED：副线与最终同轮比较

L1仅在A上合并LN1与调制；通过门且两进程三档均无可信退化后扩L13。保留原四warp Welford树、FP32加乘顺序和BF16输出；8warp从行内shared消费，普通路径省去N1/N3两个FP32全局中间量。B2仍走原路径。LN1和LN3各少336独立调制tile与一次grid，完整grid23→21，数学tile2736/2808/2736。非新的候选图调度。

L1、L13各225次有限门通过，包括B2 action、B14 action、B2 video三档KV，以及有效长度尾部、mask hole、CFG/context换序与temb不同广播形式。原23共有中间量逐位一致；Graph重放、cache与RNG通过。有限fixture不证明所有输入；B2性能本轮未测。

L13资源：108CTA×8warp，shared112384B、255寄存器/线程、local1040B（A1016B），1CTA/SM。没有动态spill计数，不把static local变化当根因。

{table}

T4-final1/2正常无插桩warm单层Graph，GPU0无其他compute进程，GPU1机器人继续；相同最终Cross KV预计算、相同真实query和有效历史。24交错样本/臂/档、每样本10 replay；表为两个独立进程中位数均值。D=原Torch静态dense历史，G=已有guarded；不是完整Python动态入口、30层、chunk或E2E。

L13相对A三档均有小幅正向，在T3和T4共四个进程方向一致；不称统计显著。相对G短快1.610%，中长慢3.602%/3.391%。LN3对L1的增量很小，T3中档约0.047%，不可宣称单独显著；最终同轮仍不见可信回退，所以保留同一简单融合模板。

## 推导、假设与UNKNOWN

- 源码可确认：LN融合减少N1/N3的FP32中间写读和两次grid；不能分离这两者各占多少收益，未实测HBM节省或动态spill。
- 假设：FFN2原bias/table已外提，可隐藏准备仅一小部分，有限重叠被布局/提交开销抵消；当前只是与结果一致的解释，没有逐项因果归因。
- UNKNOWN：完整在线输入/全部layer数值，B2新产物速度，30层/完整verifier/合法提交成本及机器人E2E；H800/5090D迁移性能。没有GPU counters，不补造带宽/occupancy实测（这里的1CTA/SM仅资源约束计算）。
- 本轮不足以支持“新warp流水优于强B14”或“投机与MegaTransformer已证协同”。

## 资源、失败与停止

本轮GPU占用窗口保守{wall/60:.3f}分钟；GPU worker生命周期{l['GPU_lifecycle_seconds']/60:.3f}分钟；CPU编译{l['CPU_compile_seconds']/60:.3f}分钟，CUDA active UNKNOWN。累计保守{l['final_cumulative_charged_seconds']/3600:.3f}/10小时。远端证据约{cleanup['storage_bytes']/2**20:.1f}MiB。全部owned退出，GPU0释放；原机器人队列不变。两次失败（TMPDIR、无效尾部审计）及真实消耗保留，无timeout。详见[RESOURCE-AND-STOP-REPORT.md](RESOURCE-AND-STOP-REPORT.md)。

## 下一轮最多三个讨论项（未自动启动）

1. 中长KV仍落后强G约3.4%–3.6%，先讨论是否继续聚焦投影/FFN primitive差距，避免将此次微小LN收益包装成主要调度创新。
2. FFN2局部准备净收益未定；若继续，需先证明更多准备覆盖能在原数值/存储边界内成立，而非扩大无界搜索。也可直接止损此路线。
3. 若补跨架构，用户当前无H100；H800可作为候选，独占/计数器权限优先。SM120的99KiB/CTA低于当前109.75KiB，5090D需另做布局适配；未授权迁移或购卡。

硬件/论文核验另见[PUBLICATION-AND-HARDWARE.md](PUBLICATION-AND-HARDWARE.md)。原LingBot-VA已正式RSS2026，论文部署RTX5880Ada，但其5v/10a与我们配置不等。

## 附：最终完整阶段表

'''+(E/'STAGE-TABLE.md').read_text()
(E/'REVIEW_RETURN.md').write_text(report)
(E/'RESULT.md').write_text('# 本轮已收口\n\n[自包含回传](REVIEW_RETURN.md)、[阶段全表](STAGE-TABLE.md)、[同轮对照](FINAL-COMPARISON.md)、[固定源码](SOURCE-COMMIT.json)、[资源](RESOURCE-AND-STOP-REPORT.md)。\n\n主线FFN2提前准备：INCONCLUSIVE，不采用。副线LN1/LN3融合：保留隔离研究L13，生产未部署。四项权限false。\n')
print('closed cumulative',l['final_cumulative_charged_seconds'],'remaining hours',l['remaining_original_seconds']/3600)
