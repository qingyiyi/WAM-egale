"""Join observed evidence, preserving partial panels and all quality failures."""
from datetime import datetime
import json
from pathlib import Path

B = Path(__file__).resolve().parent
M34 = B.parent / 'single-layer-warp-specialized-20261019'


def main():
    quality = json.loads((B / 'RESULT.json').read_text())
    timing = json.loads((B / 'ISOLATED-RESULT.json').read_text())
    snapshot = json.loads((B / 'ISOLATED-SNAPSHOT.json').read_text())
    op = json.loads((M34 / 'RESULT.json').read_text())
    terminal = json.loads((M34 / 'TERMINAL.json').read_text())
    q = (timing.get('queues') or {}).get('LONG-QUEUE-STATUS.json') or {}
    closed = q.get('status') == 'COMPLETE_ISOLATED_LONG_ENGINEERING_MEASUREMENT'
    now = datetime.now().astimezone().isoformat()
    status = 'CLOSED_PARTIAL_PRE_REGISTERED_PANEL' if closed else 'RUNNING_PARTIAL_TIMING_PANEL'
    attempts = sum(r.get('forward_attempts') or 0 for r in snapshot['rows'])
    known_phase_life = sum(p.get('runner_elapsed_wall_sec') or 0 for p in snapshot['phases'])
    origin = datetime.fromisoformat(snapshot['source_registration']['at'])
    observed = datetime.fromisoformat(snapshot['at_UTC'])
    conservative_window = max(0, 2 * (observed - origin).total_seconds())
    rep_counts = {name: {a: s['arms'][a]['N'] for a in ('T', 'C', 'P')}
                  for name, s in timing['suites'].items()}
    resource = dict(
        at=now, status=status, complete_episodes=timing['complete_rows'], original_expected_episodes=360,
        maximum_after_failed_Object_first_phase=330, failed_phase_not_retried=True,
        known_recorded_episode_target_forward_attempts=attempts,
        missing_or_not_started_rows=sum(not r['complete'] for r in snapshot['rows']),
        known_completed_runner_phase_lifecycle_seconds=known_phase_life,
        conservative_two_card_window_seconds=conservative_window,
        parent_GPU_lifecycle_cap_seconds=snapshot['source_registration']['GPU_lifecycle_cap_seconds'],
        unused_below_conservative_parent_cap_seconds=max(0, snapshot['source_registration']['GPU_lifecycle_cap_seconds'] - conservative_window),
        conservative_window_scope='both cards charged from original registration to snapshot; includes waits/failure/idle; not CUDA active or incremental M34 cost',
        CUDA_active_seconds='UNKNOWN',
        complete_and_guard_clean_phases=sum(p['complete_rows'] == 30 and p.get('guard') == 'CLEAN' for p in snapshot['phases']),
        Object_first_phase_CPU_admission_failure='0 episode / 0 teacher; elapsed CPU failure time not separately measured',
        original_work_stop=snapshot['source_registration']['absolute_work_stop'],
        original_cleanup_deadline=snapshot['source_registration']['absolute_cleanup_deadline'],
        no_automatic_retry=True, no_instance_shutdown=True,
        old_quality_permissions=timing['old_quality_permissions'],
    )
    (B / 'RESOURCE-AND-STOP-REPORT.json').write_text(json.dumps(resource, ensure_ascii=False, indent=2) + '\n')
    stop = [
        '# 资源与止损', '', f'更新时间{now}；终态/进度：{status}。', '',
        f'M35完成{timing["complete_rows"]}/360完整episode；Object首轮在CPU白名单前置检查失败，0episode/0teacher，原失败保留且不重试。仅原未启动2/3已运行，原面板最大330，不把缺30计为算法失败或缩小原预期。', '',
        f'已保存episode COUNTER的target尝试合计{attempts}；这不是把B2与B14当等成本。已结束runner phase生命周期合计{known_phase_life/3600:.4f} GPU小时口径（各卡各phase分别相加），缺少独立失败启动/排队成本，不能当全预算账。', '',
        f'更保守的原父窗口：两卡均从原T0计到本快照，合计{conservative_window/3600:.4f} GPU小时，包含已释放卡的空闲、等待、准备、失败和清理空档；低于12 GPU小时帽。未用上限约{resource["unused_below_conservative_parent_cap_seconds"]/3600:.4f}小时，不自动转授新作业。它不是CUDA active，也不与M34重叠账本再次相加为租卡费用。', '',
        '原硬限仍为10/20北京05:34:31停新、05:44:31清理；不延长。CUDA active总秒UNKNOWN，未查租卡实例计费或执行实例关机。每phase独立guard/出生时间/真实exit收据保留；最终GPU空闲与controller身份另看最后资源核对。', '',
        f'M34已关闭：{terminal["complete_block_calls"]}完整block调用（含参考/校验/计时）、0full30/episode；原数值/速度NO-GO，全部算子owned退出。详细生命周期、编译/依赖/CPU核对失败见[算子止损](../single-layer-warp-specialized-20261019/RESOURCE-AND-STOP-REPORT.md)。', '',
        '停止方向：本轮不再布局/register/grid扫参，不接新kernel到生产，不增加环境、训练、独立TEST或旧45chunk，不把PF2默认打开。新任务/样机需按后继问题重新登记，不能重置旧预算。旧FAILED/四false/retention/default A_off保留。',
    ]
    (B / 'STOP-REPORT.md').write_text('\n'.join(stop) + '\n')
    (B / 'RESOURCE-AND-STOP-REPORT.md').write_text(
        '# 本轮资源索引\n\n实际预算、失败/未重试、生命周期与硬截止见[STOP-REPORT.md](STOP-REPORT.md)，'
        '机器字段见[RESOURCE-AND-STOP-REPORT.json](RESOURCE-AND-STOP-REPORT.json)。'
        'M34独立算子账本不与M35重叠窗口相加。\n')
    facts = f'''# 事实、未知与建议

更新{now}。

OBSERVED：原四套6000完整终态齐，各臂500；成功、全部失败/长度/罚分见[QUALITY-RESULT](QUALITY-RESULT.md)。八组合max长度比均超旧1.30工程门。M34单层单kernel与warp角色已实测，原精度/速度NO-GO，600调用后收口。新计时{timing['complete_rows']}/360，状态{status}；重复仅10case/init49，不增加质量N，Object少一轮。所有臂启用原等价reset bank。

UNKNOWN：独立TEST/非劣未建立、匹配500局MLP未运行；首两S/组件失败保留。新whole kernel未准入，未测30层/chunk/episode；不存在新系统E2E速度。NCU硬件计数与全量CUDA active UNKNOWN。RoboTwin实际迁移时间/VRAM/吞吐未知；仅查官方源码/权重metadata。

HYPOTHESIS：同步单缓冲、较小GEMM tile、寄存器与粗阶段屏障共同限制新原型；不能从共享padding反事实指定bank conflict唯一根因。PF2改善某些case不等于恢复物理状态，且本面板已有第一次T之前的输入/状态差异。

RECOMMENDATION：保留可用冻结baseline；先处理Q/FFN2/attention的同输入数学契约，再考虑一个成熟SM80 tile流水原型。与已有最强cached/shared-prefix/Graph比较；不把普通缓存写成新贡献。不因开发成功率接近就开放旧质量证书或默认PF2。[日期/止损](DATES-AND-PLAN-20261020.md)。
'''
    (B / 'FACTS.md').write_text(facts)
    system = f'''# 系统证据结果

状态：M34有界原型与诊断完成、原数学/性能NO-GO；M35 {status}，{timing['complete_rows']}/360。完整原数据分别在[M34](../single-layer-warp-specialized-20261019/RESULT.md)、[M35](ISOLATED-RESULT.md)。

|诊断候选状态数|CFG/Q/线性M|本轮whole与Torch/native|缓存/强baseline状态|
|---|---|---|---|
|1|2/16/32|短中长完整block已测|本轮uncached；旧强cached/Graph有限证据另列|
|3|2/16/96|NOT_RUN，当前whole入口未准入B6|没有补造1/3/7完整矩阵|
|7|2/16/224|短中长完整block已测，仍7候选各Q16|本轮uncached；不是Q112跨候选序列|

另有B2 video Q128×三历史用于完整单层兼容验证。9 trace各1计算kernel；新原型相对native耗时5.12–11.28倍，原hidden9/9失败。175-register计时副本→198-register ldm40副本局部延迟下降6.5%–19.0%，两者理论均1 CTA/SM，仍不是原Torch精度通过。[阶段/资源](../single-layer-warp-specialized-20261019/DIAGNOSTIC-RESULT.md)、[实现问题与原语metadata](../single-layer-warp-specialized-20261019/IMPLEMENTATION-ISSUES.md)。

实际主函数172 registers/thread、59616B shared、108 CTA、LOCAL0；实测L2 requests/DRAM bytes/bank conflicts/achieved occupancy UNKNOWN。已有G1/G2或强cache的旧计数不能移填本新kernel；旧30层或tile allclose也不能继承。单kernel结构不是速度主张。

M35核心指标为ΣT_round/Σg；完整policy/chunk、全体暖episode及双方成功补充分别列[计时表](ISOLATED-RESULT.md)，[成本分解](COST-LEDGER.md)。这些作业未安装新whole kernel，其收益属于原投机实现。原观测器仍有首两S/组件FAILED；暖计时含审计/RPC/环境，不是无审计生产cold E2E。没有用target calls或接受率代替成本。

H1有限支持；H2仍缺同一冻结7-query逐B2/集中B14；H3新调度无正收益；H4在质量约束下的闭环收益未建立。A0S0/A0S1/A1S0/A1S1及1/3/7形状交互未完整执行，不计算缺失R，不声称协同。
'''
    (B / 'SYSTEM-RESULT.md').write_text(system)
    claim = '''# 联合claim—证据

|claim|当前证据|缺口/是否允许|
|---|---|---|
|四套完整开发成功率/长度|6000终态，逐任务/初态/方法和失败完整|可写描述性结果，不能称独立TEST/非劣|
|TF/PF2满足原质量门|P50/P90局部满足；max全超，MLP缺、首两S/组件失败|否，四权限false|
|投机降低有效更新成本|新同卡单worker子面板实测Σround/Σg|可限定工程成本；不是target call代替时间，也不是新whole收益|
|最终机器人E2E加速|有全体与双方成功的warm episode计时|固定10init计时、Object缺一轮，审计/曝光；正式/冷启动claim未建立|
|一层一个persistent kernel|9实际trace、真实tile/状态检查|可限定结构原型；细跨阶段流水尚无|
|整层新kernel精度/性能优|9输出原逐位失败、耗时5.12–11.28×native|否，保留负结果|
|新增候选调度超强B14与协同|当前无新增正收益，H2/交互矩阵缺|否；普通batch/cache和通用小融合不充数|
|RoboTwin可直接迁移当前TF|官方权重存在；Q32/动作16有效通道等接口不同|不能直接继承TF64/normalizer/tau/质量；只允许调研结论|

精细源码/来源与允许写法见[M34矩阵](../single-layer-warp-specialized-20261019/CLAIM-EVIDENCE-MATRIX.md)；日期和止损见[计划](DATES-AND-PLAN-20261020.md)。
'''
    (B / 'CLAIM-EVIDENCE-MATRIX.md').write_text(claim)
    report = [
        '# 10/20联合汇报：单层算子、四套LIBERO与迁移', '', f'证据汇总时间{now}。', '',
        '整层单persistent kernel已经实际运行，warp分工也已实现；当前原型的原精度和速度验收失败，不能替换冻结baseline。600调用有限研究已收口，没有把结构完成写成系统收益。', '',
        '|输入|冻结native ms|新整层ms|新/旧耗时|hidden相对L2|', '|---|---:|---:|---:|---:|',
    ]
    for r in op['cases']:
        if r['case'] in ('short-B2-action', 'short-B14-action', 'long-B14-action'):
            report.append(f'|{r["case"]}|{r["native_ms"]:.4f}|{r["whole_ms"]:.4f}|'
                          f'{r["whole_over_native_latency"]:.2f}×|{r["relative_L2"]:.6f}|')
    report += [
        '', '全部九签名、270原始计时样本、只读/写KV、RNG/参数/API与错误均保留。首轮B14 GEMM74.66%；唯一布局诊断下降6.5%–19.0%，但仅匹配失败v1，原门不变。profile/pad寄存器175/198、理论均1 CTA/SM。主要问题是成熟设备原语数值契约、同步单缓冲与粗阶段执行，不是少几个Python调用。[源码与问题](../single-layer-warp-specialized-20261019/IMPLEMENTATION-ISSUES.md)。', '',
        '|套件|Target|TF|TF+PF2|', '|---|---:|---:|---:|',
    ]
    for name, s in quality['SUITES'].items():
        counts = [s['by_arm'][a]['successes'] for a in ('T', 'C', 'P')]
        report.append('|'+name+'|'+'|'.join(f'{n}/500（{n/5:.1f}%）' for n in counts)+'|')
    report += [
        '', '6000完整局不等于独立TEST；最大成功长度比八组合均>1.30，全部失败/800罚分保留，TF/PF2没有通过原工程质量门。PF2在Long/Spatial整体成功数更低，不推荐默认开启。首两S审计还发现第一次周期T之前的相机或before-KV差异，不能把其后差异全归于PF2。[质量门](QUALITY-RESULT.md)、[首两S路径](PAIRING-DIFFERENCES.md)。', '',
        f'新工程计时状态{status}，{timing["complete_rows"]}/360；Object首轮0episode实施失败不重跑，最大330。独立server重复只用于计时，Goal/Spatial/Long各臂最多30，Object各臂20；不是新独立质量样本。', '',
        '|套件|T/C/P计时完整N|TF/PF2有效更新速度比|TF/PF2完整policy/chunk|TF/PF2双方成功暖episode|',
        '|---|---|---|---|---|',
    ]
    for name, s in timing['suites'].items():
        def pair(key):
            values = [s['pair_metrics'][a].get(key) for a in ('C', 'P')]
            return '/'.join(f'{v:.3f}×' if v is not None else 'UNKNOWN' for v in values)
        report.append('|'+name+'|'+'/'.join(str(s['arms'][a]['N']) for a in ('T', 'C', 'P'))+'|'+
            pair('valid_update_speedup')+'|'+pair('policy_cost_per_chunk_speedup')+'|'+
            pair('episode_sec_both_success_speedup')+'|')
    report += [
        '', '全体完整attempt的暖episode速度比、双方成功N、原始逐局、三/两重复的均值与SD另列[完整计时表](ISOLATED-RESULT.md)，没有删除失败或成功长尾。有效更新=Σ原action round时间/Σ合法scheduler g；不是每条树的接受率。warm包含reset/RPC/cache/环境/审计、排除env.close和落盘，冷setup单列；不是无审计生产cold E2E。新whole kernel没有装入这些作业。[账本](COST-LEDGER.md)、[边界](TIME-BOUNDARY.md)。', '',
        'RoboTwin：官方LingBot-VA权重确实存在；50唯一任务，官方适配每任务100个通过expert过滤的有效seed，一条件三臂全量15000局。其Q32/有效16动作通道、静态pool9792和三相机不同，需要新draft/normalizer/接口及质量验证；当前不适合直接复用LIBERO TF64。[官方权重](https://huggingface.co/robbyant/lingbot-va-posttrain-robotwin)、[完整迁移核查](../robotwin-migration-survey-20261019/RESULT.md)。没有启动新环境。', '',
        '下一步：先处理小M Q、FFN2与attention的同输入设备数学契约；随后只验证一条SM80兼容的tile异步流水路线，双方同用成熟cache/shared-prefix/Graph。不继续小布局搜索，不扩30层或环境，不重训来掩盖系统问题。10/22做go/no-go；10/23冻结，10/28主表，11/01 18:00中文证据包。缺口不闭合则收缩主张。[时间安排](DATES-AND-PLAN-20261020.md)。', '',
        'engineering_quality_screen=false；independent_test_complete=false；noninferiority_established=false；formal_performance_permission=false。baseline及依赖、投机源码和原tile/HW/保护文件不改。数据/权重不重复迁入，owned退出与资源硬限见[止损](STOP-REPORT.md)；原四文与短入口已归档后语义重建。GitHub实际提交号以发布收据为准。',
    ]
    (B / 'OVERNIGHT-20261020.md').write_text('\n'.join(report) + '\n')
    print(json.dumps(dict(status=status, timing_rows=timing['complete_rows'],
                          T_C_P_N=rep_counts, known_target_attempts=attempts)))


if __name__ == '__main__':
    main()
