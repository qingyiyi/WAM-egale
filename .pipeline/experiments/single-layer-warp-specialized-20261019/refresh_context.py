"""Rebuild short derived navigation after preserved M34-complete archive."""
from datetime import datetime
import json
from pathlib import Path
import re

B = Path(__file__).resolve().parent
ROOT = B.parents[2]
PIPE = ROOT / '.pipeline'
CYCLE = PIPE / 'cycles/wam-research-20261016'


def main():
    assert (CYCLE / 'history/M34-complete-20261020/PROGRESS.md').exists()
    now = datetime.now().astimezone().isoformat()
    timing = json.loads((B.parent / 'libero-four-suite-performance-20261019/ISOLATED-RESULT.json').read_text())
    long_queue = (timing.get('queues') or {}).get('LONG-QUEUE-STATUS.json') or {}
    closed = long_queue.get('status') == 'COMPLETE_ISOLATED_LONG_ENGINEERING_MEASUREMENT'
    count = timing['complete_rows']
    phase = '已收口，原面板PARTIAL' if closed else '运行中'
    long_note = 'Long三phase已全部终态并清理；四套本轮没有继续排队的GPU作业。' if closed else 'Long按原单worker队列自动接续，实际终态以表为准。'
    next_note = '本轮已登记测量和有界样机均已终态；下一步交联合报告并讨论缺口，不自动追加GPU作业。' if closed else '先完成M35终态/清理并同步main；M34本轮已止损。'
    progress = f'''# 当前研究进度

更新：{now}。当前范围只有一层；原四套计算和新计时重复分开。[短入口](../../PROGRESS.compact.md)。

## M17 — 四套既定评估：计算完成，审计未放行

6000完整episode，四套各T/C/P500：Long483/484/473、Goal463/464/467、Spatial472/466/464、Object497/498/499。[成功/长度/罚分及门](../../experiments/libero-four-suite-performance-20261019/QUALITY-RESULT.md)。TF/PF2全部八组合P50/P90在旧界内，max全部超过1.30；首两S严格配对FAILED、训练/DEV曝光、组件审计与全部失败保留。无匹配500局MLP；不是独立TEST/非劣。GPU计算已清理，科学门仍未完成。

## M31 — 独立静态tile子图：completed

[原结果](../../experiments/transformer-tile-subgraph-20261019/RESULT.md)：9block/72状态和独立54状态有限通过。不能继承给M34新算术。

## M32 — 阶段内persistent：completed

[原结果](../../experiments/persistent-tile-remote-20261019/RESULT.md)：9block/90状态/48原语有限通过；378样本中B14比静态慢4.42%–6.26%，A/A最大10.07%。attention/FFN1阶段内常驻，其他GEMM仍host cuBLASLt；不满足整层单kernel目标。731block、77.927秒，owned退出。

## M33 — 退化profile：completed

[归因](../../experiments/single-block-persistent-profile-20261019/RESULT.md)：长B14两路径各34kernel；增量207.072us由三替换组件207.2425us解释。216 CTA反事实把短中长退化降到0.726%/0.474%/0.179%，仍无正收益。NCU权限失败、计数UNKNOWN，两次扩展身份实施失败保留。293block/81.856秒、owned退出。

## M34 — 整层单kernel与warp分工：completed，验收NO-GO

[主源码](../../../算子研究/wam_ops/optimized/csrc/whole_layer.cu)与[入口说明](../../../算子研究/wam_ops/optimized/WHOLE_LAYER.md)。全部self/cross、投影、FFN、Norm/RoPE/调制/残差/KV在一次cooperative launch内，8warp分调度/加载/计算/写回；粗阶段grid barrier，没有跨阶段细流水。9真实签名各trace1kernel、tile/输入/RNG/参数及metadata通过；9最终hidden逐位FAIL，相对L2约0.13%–0.56%，18写序列中9数值FAIL。

[两进程270样本](../../experiments/single-layer-warp-specialized-20261019/RESULT.md)：耗时为native的5.12–11.28倍；不是有效加速。172 registers/thread、59616B shared、理论1 CTA/SM；非实测occupancy。[阶段诊断](../../experiments/single-layer-warp-specialized-20261019/DIAGNOSTIC-RESULT.md)B14 GEMM74.66%，唯一ldm40副本下降6.5%–19.0%，仅match失败v1、仍未准入。[精度边界](../../experiments/single-layer-warp-specialized-20261019/PRECISION-CUTS.md)独立定位Q、FFN2和video attention差异，不以splitK单因解释。

累计600/600block、0full30/episode，01:44:39北京清理。编译/依赖/CPU核对失败保留；本机88及远端86已部署保护项不变，远端未迁入state/rules的原88检查false不改。[资源/止损](../../experiments/single-layer-warp-specialized-20261019/RESOURCE-AND-STOP-REPORT.md)。不扩扫参、不集成、不降低门。

## M35 — 新同卡工程计时：{phase}

[实际快照](../../experiments/libero-four-suite-performance-20261019/ISOLATED-RESULT.md)：{count}/360完整局，固定10task/init49×三臂×3独立server重复；计时重复不增加独立质量N。Goal/Spatial各90，Object60；Object1误送旧Goal/Spatial白名单在0episode/0teacher失败，保留不重试，只接续原未启动2/3，最大330/360。{long_note}

GPU0 Goal→Object，GPU1 Spatial→Long；同case对照同卡，每卡单worker、server/client2线程。05:34:31停新、05:44:31清理（10/20），1452000尝试/3GiB/12 GPU生命周期小时帽不延期。首两S组件FAILED仍列出，计时只作未准入候选成本观察；不开放旧45chunk。warm含reset/policy/cache/env，不含env.close/落盘；冷setup另列。[边界](../../experiments/libero-four-suite-performance-20261019/TIME-BOUNDARY.md)。

## M36 — RoboTwin只读调研：completed

[官方核查](../../experiments/robotwin-migration-survey-20261019/RESULT.md)：有LingBot-VA官方权重，50唯一任务，原适配100有效seed/task（先expert过滤），T/C/P全套15000局/condition。Q32/有效16动作通道、pool静态9792，当前TF64/API/阈值不能直接继承。没有装环境、下权重或启动实验。

## 下一步及保护

{next_note}若继续新算子研究，先讨论匹配设备GEMM/attention契约的可行原语和强cached B14，另登记有界作业；不能把普通cache收益叫协同创新。[claim矩阵](../../experiments/single-layer-warp-specialized-20261019/CLAIM-EVIDENCE-MATRIX.md)、[日期](../../experiments/libero-four-suite-performance-20261019/DATES-AND-PLAN-20261020.md)。

四权限false；旧FAILED/retention/NO_GO/paused72/96、MLP0、22缺收据、default A_off不改。baseline及依赖/投机源/原tile与HW/state/cycle/rules保护；仅optimized新实现，不重置/旧sync/升级，不建缺失cycle.yaml。远端只~/cxz，AutoDL不再SSH/关机UNKNOWN；不杀他人、不改服务/时钟。

M17旧10/21 16:56:31停新、17:11:31清理、18723000尝试/15GiB保留，不由M35重置。无新训练/120/独立TEST/第二环境/30层授权。仅main公开核对源码/HW/小证据，无模型/PT/KV/DSO/凭据。10/22 go-no-go、10/23冻结、10/28主表、11/01 18:00中文包；正文由用户飞书整理。之前状态已归档[history](history/M34-complete-20261020/PROGRESS.md)。
'''
    plan = f'''---
kind: plan
cycle: wam-research-20261016
status: active
updated: '{now}'
progress: PROGRESS.md
execution: EXECUTION.md
steps:
  - id: M17
    status: in_progress
  - id: M31
    status: completed
  - id: M32
    status: completed
  - id: M33
    status: completed
  - id: M34
    status: completed
  - id: M35
    status: {'completed' if closed else 'in_progress'}
  - id: M36
    status: completed
---
# 当前有界计划

M17计算齐但质量审计未放行；M31–M33按原结果完成。M34单层单kernel原型/有限诊断交付完成，原精度与速度NO-GO；completed不代表科学验收通过。M35为{phase}，原360最多330，失败不重跑；M36只读调研完成。逐项证据见[完整PROGRESS](PROGRESS.md)，之前计划保留[归档](history/M34-complete-20261020/PLAN.md)。

下一步：{next_note}实际工程成本/边界、缺项与负结果统一汇报，main只公开核对源码/HW/小证据。M34累计600调用后关闭，不借旧余额自动开新实现/训练/TEST。原05:34停新/05:44清理不延期。

新算子研究需先解决Q/FFN2/attention设备数值契约与强cached B14关系；当前不追加GPU网格搜索。10/22若关键证据不足就收缩主张。保护文件、质量四false、范围与资源限制按PROGRESS和约束记忆，不覆盖state/cycle/rules。
'''
    execution = f'''# 最近执行

## {now}：M34有界收口与M35进度核对

M34整层kernel在9真实签名各一次计算launch；原输出9/9逐位失败，成本为冻结native的5.12–11.28倍。18+65调用的阶段/唯一布局诊断完成，B14 GEMM74.66%；ldm40局部下降6.5%–19.0%，仍原精度FAIL。累计600调用、0full30/episode，全部算子owned退出；本机88/远端86已部署保护项最终不变，远端state/rules未迁入的原false保留。[终态](../../experiments/single-layer-warp-specialized-20261019/TERMINAL.json)。

M35当前{count}/360，Object首轮0episode/0teacher的CPU白名单失败保留，不自动重试；仅原未启动2/3已完成。Long队列状态{long_queue.get('status', 'UNKNOWN')}；完整episode/时间覆盖/组件失败从终态和COUNTER统计，不按进程exit0判机器人成功。四false不改。

当前导航按语义重建，旧四文与短入口/索引保存到history/M34-complete-20261020；没有旧sync、升级或覆盖保护文件。先前执行与失败见[归档](history/M34-complete-20261020/EXECUTION.md)。
'''
    discussion = '''# 当前讨论与授权

最新用户要求现在直接实现：整个单层Transformer在一个persistent kernel内，把全部算子切tile交给职责不同的warp，借鉴MegaMoE；明早汇报实测速度、精度与问题。只一层，不做30层大融合。baseline及共享依赖冻结，仅optimized新入口；投机代码/配置与原tile不改。

用户允许四套LIBERO速度统一远端测，替代旧“最终E2E必须本机”的地点条款；每卡计时单worker且无观察到争用，M34 GPU验证等对应计时卡清理，CPU实现可并行。本机可用自然空闲窗口先核对精度；不杀他人。AutoDL已归还不再连接，BAAI所有写入限~/cxz。

当前M34结构实测成立但精度/性能NO-GO，已按600调用有界诊断收口；没有授权放宽原门。M35原360面板因Object1实施失败最多330，原失败保留、不自动重跑或延长截止；原未启动2/3可以按既定预算接续。RoboTwin仅调研，没有新增实验/安装/权重下载权限。

四质量权限保持false，旧FAILED/retention/NO_GO/paused72/96、MLP0、22缺收据/default A_off保留。不能用成功子集、调用减少、match-v1、单kernel结构或普通batch/cache宣称质量与协同加速。新训练/120/独立TEST/受阻45chunk不自动启动。

只main公开核对源码/HW/小证据，模型/PT/KV/图像/DSO/工具链/连接配置/凭据不公开。中文正文逐段交用户飞书，不建本地初稿。10/22 go/no-go、10/23冻结、10/28表、11/01 18:00老师中文证据包；官方投稿日期见日期报告，作者profile仍UNKNOWN。

旧讨论原文保留[归档](history/M34-complete-20261020/DISCUSSION-SUMMARY.md)；保护state/cycle/rules、不升级或旧sync。当前进度以PROGRESS与实际终态为准。
'''
    compact = f'''# 当前短入口

更新：{now}。当前Cycle：[wam-research-20261016](cycles/wam-research-20261016/PROGRESS.md)。先[约束记忆](memory/INDEX.md)，再读当前四文；原长记录/trace按需，不展开全部旧索引。

## 已观察结果

四套原6000完整episode齐：每套10task×50init×T/C/P，成功数Long483/484/473、Goal463/464/467、Spatial472/466/464、Object497/498/499，各臂500。[完整质量/门](experiments/libero-four-suite-performance-20261019/QUALITY-RESULT.md)、[原逐局与成本](experiments/libero-four-suite-performance-20261019/RESULT.md)。八组合P50/P90都在旧长度界内，但max都>1.30；失败/800罚分、首两S严格配对FAILED、组件失败和训练/DEV曝光保留，不是独立TEST/非劣。无匹配500局MLP。

M34已完成整层单kernel/warp分工原型与有界诊断，**原精度和速度NO-GO**。[完整CUDA](../算子研究/wam_ops/optimized/csrc/whole_layer.cu)、[说明](../算子研究/wam_ops/optimized/WHOLE_LAYER.md)。全部self/cross、投影、FFN、Norm/RoPE/调制/残差/KV都在一次cooperative计算launch，8warp按调度/两加载/四计算/写回；粗阶段grid barrier，全局中间量，没有细跨阶段流水。

[9真实签名](experiments/single-layer-warp-specialized-20261019/RESULT.md)各1kernel、tile/输入/RNG/参数和metadata通过；最终hidden9/9逐位FAIL，相对L2约0.13%–0.56%，只读KV9通过、18写序列9数值FAIL。两进程270样本，新耗时为native的5.12–11.28倍，不能称加速。172 registers/thread、59616B shared、理论1 CTA/SM，非achieved occupancy。

[阶段诊断](experiments/single-layer-warp-specialized-20261019/DIAGNOSTIC-RESULT.md)三B14 GEMM74.66%；唯一ldm40副本局部延迟下降6.5%–19.0%，仅匹配失败v1，原门不变。[精度定位](experiments/single-layer-warp-specialized-20261019/PRECISION-CUTS.md)：同输入Q/FFN2及video attention边界有差异，cross K/V通过；不把splitK指定为唯一根因。累计600/600block、0full30/episode，01:44:39北京清理，owned退出。原编译/依赖/CPU核对失败及远端未迁入state/rules的88项检查false保留；本机88、远端86已部署项最终不变。[资源/止损](experiments/single-layer-warp-specialized-20261019/RESOURCE-AND-STOP-REPORT.md)。

## {'资源终态与硬截止' if closed else '正在运行与硬截止'}

M35新同卡工程计时{phase}，当前{count}/360完整局，固定10task/init49×三臂×3独立server进程重复。Goal/Spatial各90，Object60；Object1误入旧Goal/Spatial白名单，0局0teacher，保留不重试，仅接续原未启动2/3，最大330。{long_note}逐arm数见[新计时结果](experiments/libero-four-suite-performance-20261019/ISOLATED-RESULT.md)。计时重复不增加独立质量N。

GPU0 Goal→Object、GPU1 Spatial→Long，每卡单worker/server与client各2线程，同case同卡；05:34:31停新、05:44:31清理（10/20），12 GPU生命周期h/1452000尝试/3GiB，不延期。计时仍有首两S组件FAILED，只作未准入算法工程成本观察，不是旧正式入口放行。核心ΣT_round/Σg；暖episode含reset/policy/cache/env，不含env.close/落盘，冷setup另列，client_total不叫纯E2E。[计时边界](experiments/libero-four-suite-performance-20261019/TIME-BOUNDARY.md)。

## 接口与已有强基线

TF64/每套选定权重、normalizer、tau=.007839680526450805、absΔσ×L2、BF16方向/FP32评分、K2B2/bonus/prefix_then_cost冻结。PF2=SST完整policy请求，T50更新/51 action调用含终端；reset/cache不计，不是bonus或状态恢复。所有臂同用512-token BF16 reset bank，首次构建/冷启动另列。actual796与失败罚分800分开。

D3072/H24/d128/FFN14336，action Q16、GEMM M32→224已有实际aten::addmm证据；attention实例仍Q16。action KV288/1152/2160，video本批272/1136/2144，cross512；candidate private KV不共享。[旧强cache/Graph与FFN组件](experiments/operator-profile-device-20261019/RESULT.md)已有0.7%–3.2%有限收益，不作新调度创新。M34未接成熟cache强路径。

M31静态子图、M32阶段内persistent、M33退化profile旧结果保留；M32新增persistent更慢，M33调216 CTA仍无正收益，NCU权限失败/硬件计数UNKNOWN，不借旧有限精度给新算术签证。[主张矩阵](experiments/single-layer-warp-specialized-20261019/CLAIM-EVIDENCE-MATRIX.md)：H1有限支持；H2同冻结7-query对照缺；H3无新增调度正收益；H4质量约束闭环证据未完成。

## 下一步与范围

{next_note}M34本轮停止GPU扫参；若继续，先论证设备GEMM/attention原契约与成熟tile流水原语，另登记预算。shape×engine的1/3/7完整矩阵和算法/系统交互缺口不能补写为完成；普通batch/cache不算MegaTransformer创新。

M36[RoboTwin调研完成](experiments/robotwin-migration-survey-20261019/RESULT.md)：官方LingBot-VA权重/50唯一任务，100有效seed/task，三臂15000局/condition。Q32/16有效动作通道/pool静态9792需新draft/cache接口，不继承LIBERO阈值；0第二环境实验。CUDA仍私有12.4/Torch2.9 cu126，远端driver535不升级CUDA13/系统。

## 保护与日历

engineering_quality_screen=false；independent_test_complete=false；noninferiority_established=false；formal_performance_permission=false。旧FAILED/retention/NO_GO/paused72/96、MLP0、22缺收据/default A_off不改。

baseline及依赖冻结，仅optimized变化；投机源/配置/运行依赖、原tile/HW、state/cycle/rules保护，不建缺失cycle.yaml、不升级/旧sync。BAAI仅~/cxz；AutoDL不再SSH/关机UNKNOWN。用户允许性能统一远端，替代旧地点限制；不杀他人/改服务/时钟。无自动新训练/120/独立TEST/旧45chunk/第二环境/30层大融合。

M17原10/21 16:56:31停新、17:11:31清理、18723000尝试/15GiB不重置。只main公开审查源码/HW/小证据，无权重/PT/KV/DSO/凭据。10/22 go/no-go、10/23冻结、10/28主表、11/01 18:00中文包；正文由用户飞书整理。官方注册/正文/补充北京11/11、11/17、11/24各19:59，未单列abstract截止，作者profile UNKNOWN。[日期与三情形](experiments/libero-four-suite-performance-20261019/DATES-AND-PLAN-20261020.md)。旧方案见Cycle history，不由历史“下一步”重启任务。
'''
    assert len(compact.encode()) <= 8192, len(compact.encode())
    ids = re.findall(r'^  - id: (M\d+)$', plan, re.M)
    progress_ids = re.findall(r'^## (M\d+) —', progress, re.M)
    assert ids == progress_ids, (ids, progress_ids)
    for name, text in [('PLAN.md', plan), ('PROGRESS.md', progress),
                       ('EXECUTION.md', execution), ('DISCUSSION-SUMMARY.md', discussion)]:
        (CYCLE / name).write_text(text)
    (PIPE / 'PROGRESS.compact.md').write_text(compact)
    index = (PIPE / 'INDEX.md').read_text()
    index = index.replace('[当前整体整理]', '[10/19历史阶段整理]')
    start = index.index('当前算子：')
    end = index.index('\n## 两条研究路线', start)
    index = index[:start] + ('当前算子：[M34整层单kernel实际结果](experiments/single-layer-warp-specialized-20261019/RESULT.md)和'
             '[有界阶段诊断](experiments/single-layer-warp-specialized-20261019/DIAGNOSTIC-RESULT.md)已完成，'
             '结构成立、原精度/速度NO-GO、600调用后清理；不是已准入实现。'
             'M35同卡工程计时终态见[当前结果](experiments/libero-four-suite-performance-20261019/ISOLATED-RESULT.md)，'
             'Object首轮实施失败不重跑，原面板最多330/360。旧四false/FAILED保留。\n') + index[end:]
    index = re.sub(r'更新时间：[^。]+。', f'更新时间：{now}。', index, count=1)
    (PIPE / 'INDEX.md').write_text(index)
    print(json.dumps(dict(compact_bytes=len(compact.encode()), timing_rows=count,
                          M35_closed=closed, Plan_Progress_IDs=ids), ensure_ascii=False))


if __name__ == '__main__':
    main()
