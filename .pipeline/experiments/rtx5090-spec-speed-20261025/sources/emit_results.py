"""Render a final report only after verified return and remote directory removal."""
from pathlib import Path
from datetime import datetime
import csv, json

P = Path(__file__).resolve().parent
d = json.loads((P/'RESULTS.json').read_text())
ack = json.loads((P/'FINAL-RETURN-ACK.json').read_text())
deleted = json.loads((P/'FINAL-REMOTE-DIRECTORY-DELETION.json').read_text())
assert not deleted['root_exists_after']
assert all(value is False for value in d['permissions'].values())
fmt = lambda x: 'UNKNOWN' if x is None else f'{x:.3f}'
names = dict(long='LIBERO Long', goal='LIBERO Goal', spatial='LIBERO Spatial', object='LIBERO Object', robotwin='RoboTwin')
text = ['# RTX5090投机侧完整自然任务成本结果', '',
        '本轮使用原冻结Target、TF和同权重TF+PF2，完整512-token BF16 reset bank，单模型worker、eager/no CUDA Graph；未接入自研单层Transformer。原始字段、失败与旧质量门全部保留。', '',
        '## 完整配对面板', '',
        '| 环境 | 完整三臂组N | 成功T/TF/PF2 | TF动作成本加速 | PF2动作成本加速 | TF完整policy/chunk | PF2完整policy/chunk | TF暖任务E2E | PF2暖任务E2E |',
        '|---|---:|---|---:|---:|---:|---:|---:|---:|']
for name in ['long', 'goal', 'spatial', 'object', 'robotwin']:
    if name not in d['suites']: continue
    s = d['suites'][name];c=s['comparisons']['C'];p=s['comparisons']['P']
    success='/'.join(str(s['arms'][a]['success']) for a in 'TCP')
    text.append(f"| {names[name]} | {s['N']} | {success} | {fmt(c['action_update_speedup'])}× | {fmt(p['action_update_speedup'])}× | {fmt(c['policy_chunk_speedup'])}× | {fmt(p['policy_chunk_speedup'])}× | {fmt(c['all_complete_warm_episode_speedup'])}× | {fmt(p['all_complete_warm_episode_speedup'])}× |")
text += ['', '动作主成本为Σ完整action耗时/Σ合法scheduler更新g；逐轮host计时加准备/终端/排空残差与路由开销闭合到该完整action，避免漏算异步尾部。完整policy/chunk是自然任务中的平均policy耗时，不是固定KV微基准。加速比均为同套Target成本除以相应投机成本。', '',
         '## 全部尝试与成功长度', '',
         '| 环境/方法 | 全部完整尝试客户端生命周期E2E | 双方成功N | 双方成功生命周期E2E | 有效环境步数比P50/P90/max | 原登记失败罚分相对T变化 |',
         '|---|---:|---:|---:|---|---:|']
for name in ['long', 'goal', 'spatial', 'object', 'robotwin']:
    if name not in d['suites']: continue
    s=d['suites'][name]
    for arm,label in [('C','TF'),('P','PF2')]:
        r=s['comparisons'][arm];ratios='/'.join(fmt(r[k]) for k in ['length_ratio_P50','length_ratio_P90','length_ratio_max'])
        text.append(f"| {names[name]}/{label} | {fmt(r['all_complete_client_lifecycle_speedup'])}× | {r['both_success']} | {fmt(r['both_success_client_lifecycle_speedup'])}× | {ratios} | {r['paired_failure_penalty_delta']:+d} |")
text += ['', '计入全部失败耗时的成功吞吐：分子为成功数，分母为所有完整尝试client生命周期之和；不包含迁移、冷装载和局间间隙，因此不是整个租卡账单GPU小时。', '',
         '| 环境 | Target成功/测得小时 | TF成功/测得小时 | PF2成功/测得小时 | TF/T吞吐比 | PF2/T吞吐比 |',
         '|---|---:|---:|---:|---:|---:|']
for name in ['long', 'goal', 'spatial', 'object', 'robotwin']:
    if name not in d['suites']: continue
    s=d['suites'][name];rates=[fmt(s['arms'][a]['successes_per_measured_client_hour_including_failures']) for a in 'TCP']
    text.append(f"| {names[name]} | {' | '.join(rates)} | {fmt(s['comparisons']['C']['all_attempt_success_throughput_speedup'])}× | {fmt(s['comparisons']['P']['all_attempt_success_throughput_speedup'])}× |")
text += ['', '自然任务时间账本（各项占全部完整尝试client生命周期百分比）：', '',
         '| 环境/方法 | action | policy其余 | cache RPC | reset RPC | 暖期其余 | 启停等 |',
         '|---|---:|---:|---:|---:|---:|---:|']
phase_fields=['action_seconds','policy_nonaction_seconds','cache_seconds','reset_seconds','warm_other_seconds','startup_close_other_seconds']
for name in ['long', 'goal', 'spatial', 'object', 'robotwin']:
    if name not in d['suites']: continue
    for arm,label in [('T','Target'),('C','TF'),('P','PF2')]:
        r=d['suites'][name]['arms'][arm];total=r['client_lifecycle_seconds']
        assert abs(sum(r[k] for k in phase_fields)-total)<1e-6
        values=[f'{r[k]/total*100:.2f}%' if total else 'UNKNOWN' for k in phase_fields]
        text.append(f"| {names[name]}/{label} | {' | '.join(values)} |")
text += ['', '六项以未舍入数据闭合到client生命周期。policy其余包括video、VAE/输入处理及该RPC的其余工作；暖期其余包含环境/规划/渲染/动作处理与请求外的client工作，不应直接称纯物理仿真时间；启停等包含进程/场景初始化和关闭及外层等待。', '',
         '客户端生命周期包括新client启动、导入/场景初始化、reset、policy/cache、环境执行和关闭，模型server已常驻；首次模型装载和权重迁移另列资源成本。LIBERO暖口径沿原client（环境建立后到自然终态）；RoboTwin暖口径为首次reset RPC到首次env.close。不同环境的暖计时边界不合并成一个总加速比。全部完整尝试包含算法失败，成功子集只是补充。分位数使用(N-1)q线性插值。', '',
         'RoboTwin保留原各任务horizon，微波炉1500而非LIBERO800；有效环境步与末尾多提交命令分列。旧登记罚分口径不改，另给基于有效环境步的罚分。', '',
         '## 证据限制与采用决定', '',
         '- 已暴露DEV的有限计时面板；不是每套500初态或独立TEST，也不是严格非劣证明。Long两条已执行PARTIAL保留且未重跑，共同三臂只有8组；旧默认FSDP的单条Target不混入强noFSDP主对照。',
         '- RoboTwin六组中篮子新增失败、TF订书机成功长尾，不能仅报动作加速。五case在第一个PF2刷新之前的第二个S已有差异；六case首个S保存分数/前16命令相同，具体后续分歧根因UNKNOWN。RoboTwin比较是描述性实际执行成本，不支持PF2纯单因素因果结论。',
         '- 有限路由/计数/RNG、保存的分数/命令检查见AUDIT-RESULT.json；完整中间观测/KV/velocity未全部持久化，不能声称全状态逐位等价。不同GPU上的历史质量结论不能直接继承。',
         '- 单GPU单模型worker，逐局保存GPU上下文快照；部分阶段在独立CPU核受控预传下一套资产，网络/磁盘并非全程空闲。预传生命周期另存PREFETCH记录，本轮是工程成本观察，不冒充无后台IO且多进程重复的正式论文计时。',
         '- 不改变TF/PF2生产默认、阈值、训练方法或质量权限，不把target calls减少替代速度。下一步先完成原A100质量队列，再针对RoboTwin分叉前差异做有界定位；本轮未自动追加训练或机器人重跑。', '',
         '四项均保持false：engineering_quality_screen、independent_test_complete、noninferiority_established、formal_performance_permission。', '',
         '## 回传与清理', '',
         f"原始结果/源码共{len(ack['files'])}文件、{ack['total_bytes']/2**20:.2f}MiB，native rsync完整性、不可变文件尺寸清单与本地fsync完成后，已删除整个本轮5090临时根目录。删除表观字节{deleted['deleted_apparent_bytes']/2**30:.3f}GiB；原算子环境保留，未关闭实例。进程身份清理与GPU上下文见FINAL-OWNED-PROCESS-CLEANUP.json。", '',
         '固定结果入口：RESULTS.json、EPISODES.csv、AUDIT-RESULT.json；原始数据raw/；逐套回传和删除收据*-RETURN-ACK.json、*-WEIGHT-DELETION.json；最终FINAL-RETURN-ACK.json与FINAL-REMOTE-DIRECTORY-DELETION.json。']
(P/'PERFORMANCE-RESULT.md').write_text('\n'.join(text)+'\n')
print('PERFORMANCE-RESULT.md generated from completed immutable evidence and cleanup receipts')
