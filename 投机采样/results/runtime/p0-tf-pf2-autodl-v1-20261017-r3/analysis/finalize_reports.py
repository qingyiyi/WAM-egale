"""Bounded CPU-only synthesis of preserved, already completed PF2 evidence."""
from pathlib import Path
from datetime import datetime, timezone
import csv
import io
import json
import os
import signal
import time

OWN = Path(__file__).resolve().parent
JOB = OWN.parent
RAWBASE = JOB / 'remote-archive/投机采样/results/runtime'
RAW = RAWBASE / JOB.name
OUT = OWN / 'closure-v1'


def read(p):
    return json.loads(Path(p).read_text())


def write(name, data):
    raw = data if isinstance(data, str) else json.dumps(data, ensure_ascii=False, indent=2) + '\n'
    with (OUT / name).open('x') as f:
        f.write(raw)


def dt(s):
    return datetime.fromisoformat(s).timestamp()


def main():
    assert not OUT.exists() and len(os.sched_getaffinity(0)) == 1 and os.environ['CUDA_VISIBLE_DEVICES'] == ''
    OUT.mkdir()
    started = time.monotonic()
    write('PREREGISTRATION.json', dict(goal=__doc__, CPU=1, GPU=False, wall_cap_sec=120,
        storage_cap_bytes=8 * 2**20, source_artifact_root=str(RAW), no_source_edits=True))
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('CPU synthesis cap')))
    signal.alarm(120)
    result = read(JOB / 'report-v1/RESULT.json')
    gates = read(OWN / 'result-v1/INVESTMENT-RESULT.json')
    plan = read(JOB / 'plan.json')
    probe = read(OWN / 'POSTRUN-RESOURCE-OBSERVATION.json')
    assert result['complete_episodes'] == 12 and result['panel_complete'] is True
    assert gates['pilot_investment_decision'] == 'NO_GO' and set(result['permissions'].values()) == {False}
    assert probe['registered_live'] == []
    summaries, servers = [], []
    for i in (1, 2, 3):
        folder = RAWBASE / f'p0-tf-pf2-autodl-v1-20261017-r{i}'
        sup = read(folder / 'SUPERVISOR-RESULT.json')
        cleanup = read(folder / 'FINAL-OWNED-CLEANUP.json')
        assert cleanup['all_owned_groups_empty'] is True
        summaries.append(dict(attempt=i, status=sup['status'], elapsed_wall_sec=sup['elapsed_wall_sec'],
            complete_episodes=sup['complete_episodes'], original_authorization_elapsed_sec=sup['wall_sec_since_original_authorization'],
            all_owned_groups_empty=True))
        for r in cleanup['exits']:
            if not r['role'].startswith('server-'):
                continue
            pin = read(folder / f"OWNED-{r['pid']}.json")
            servers.append(dict(attempt=i, pid=r['pid'], birth_ticks=pin['birth_ticks'], role=r['role'],
                registered_at=pin['registered_at'], reaped_at=r['at'], returncode=r['returncode'],
                provisioned_worker_lifecycle_sec=dt(r['at']) - dt(pin['registered_at'])))
    counters = [read(RAW / 'episodes' / f"pair-{r['pair_id']:02d}" / r['arm'] / 'COUNTER.json') for r in result['rows']]
    reset = [dict(pair_id=c['pair_id'], arm=c['arm'], sec=r['original_infer_host_seconds'])
        for c in counters for r in c['requests'] if r['kind'] == 'reset']
    assert len(reset) == 12
    request_cost = {a: {k: sum(r['original_infer_host_seconds'] for c in counters if c['arm'] == a
        for r in c['requests'] if r['kind'] == k) for k in ('reset', 'cache', 'policy')} for a in ('C', 'P')}
    resource = dict(attempts=summaries, provisioned_server_lifecycles=servers,
        provisioned_server_lifecycle_hours=sum(s['provisioned_worker_lifecycle_sec'] for s in servers) / 3600,
        whole_model_CUDA_active_time='UNKNOWN', original_authorization=plan['authorization_origin'],
        absolute_cleanup_deadline=plan['absolute_cleanup_deadline'], no_new_work_after_panel=True,
        instrumented_episode_target_forwards=sum(r['target_forward_attempts'] for r in result['rows']),
        pre_environment_client_setup_failures=2, missing_r2_counter_conservative_debit=8000,
        non_episode_unmeasured_forward_reserve=1000,
        conservative_forward_book=16054 + 8000 + 1000, forward_cap=49000,
        forward_balance_after_conservative_reserves=49000 - 16054 - 8000 - 1000,
        all_global_target_forward_calls='UNKNOWN_OUTSIDE_INSTRUMENTED_RECORDER',
        per_episode_max_observed=max(r['target_forward_attempts'] for r in result['rows']),
        reset_requests=reset, disjoint_native_infer_request_host_seconds=request_cost,
        complete_reset_bank_enabled_both_workers=all(read(RAW / f'server-worker{i}/READY.json')['complete_text_bank'] for i in (0, 1)),
        collected_raw_evidence=read(OWN / 'COLLECTION-RECEIPT.json'), postrun_resource_observation=probe,
        instance_shutdown='NOT_REQUESTED_NOT_EXECUTED', permissions=result['permissions'])
    write('RESOURCE-DETAILS.json', resource)
    rows = ['|pair|task/init|分层|连续 C：终态/实际命令|PF2 P：终态/实际命令|', '|---|---|---|---|---|']
    by = {(r['pair_id'], r['arm']): r for r in result['rows']}
    for pair in result['pairs']:
        c, p = by[(pair['pair_id'], 'C')], by[(pair['pair_id'], 'P')]
        text = lambda r: ('成功' if r['success'] else '失败') + '/' + str(r['actual_env_steps'])
        rows.append(f"|{pair['pair_id']}|{pair['task']}/{pair['init']}|{pair['stratum']}|{text(c)}|{text(p)}|")
    table = '\n'.join(rows)
    quality = f'''# 冻结 TF_ON：连续与 PF2 的有限 DEV 结论

OBSERVED：固定科学快照4f04b35，实际独立执行源码9acc37cee9ae9b62b8311f1f8dbb01bffa37a86a。六对十二局全部完整，初态物理/controller/RNG/双相机保存组件6/6严格匹配；原始阈值、权重、normalizer与树未改。此为暴露过的风险富集DEV，不是独立TEST。

{table}

C成功3/6，P成功5/6；风险0/3→2/3，对照3/3→3/3。本轮真实rescued=2、lost_success=0、双方成功3、双方失败1。没有用历史C替代本次C，也没有原Target/MLP新实验臂。实际命令总数C3110/P2166；失败800罚分分别3122/2170。风险罚分2400→1305，对照722→865；各失败实际796命令，罚分800，未补造四命令。

共同成功3对的P/C长度比：P50=1.0042735，P90=1.4470085，max=1.5576923；分位数为线性index=(n−1)q。task1/init36从260变405，是明确的长度反例。风险救回不能抵消控制组罚分、P90和max失败；task4/init26的PF2仍失败，P风险全成功门也失败。

**登记投入结论：NO_GO。** 详见../result-v1/INVESTMENT-RESULT.json的逐门重算；两项成本门因可比性未知保持UNKNOWN，不能改成PASS。

OBSERVED：探索性ΣT_round/Σg为C=0.040679905、P=0.052514468秒/合法更新；P/C约1.2909。这里包含实际draft/anchor/verify/pack/接受/bonus/末尾action工作。内部g不是环境步。全体完整policy/chunk为3.612273→4.192598秒；共同成功的policy总成本165.899302→226.989656秒（P/C=1.368238）。全体policy总成本711.617718→582.771179秒，包含失败，PF2少了两条跑满失败。B2/B14 forward分别保留typed账本，不将8643→7411调用数称为加速。

UNKNOWN：稀疏资源采样包含无法映射到容器出生身份的GPU PID，成本连续可比性未建立；上述是带审计AutoDL观察，正式E2E speedup、相对原Target速度、总体LIBERO成功率、非劣性、完整GPU活跃时间均UNKNOWN。

HYPOTHESIS：PF2可能缓解部分持续投机风险；它同时改变draft提交频率和B2/B14 teacher路径比例，当前实验不能单独证明误差积累或某一数值算子是根因。

RECOMMENDATION：本轮结束，不把连续或PF2设为已验收的最终TF方案，不上全量训练/120/TEST/正式性能，不搜PF3/4、不改tau。TF64结构/API继续作为研究接口。下一项应是另行登记的教师路径因果诊断，而非无依据追加训练。

engineering_quality_screen=false；independent_test_complete=false；noninferiority_established=false；formal_performance_permission=false。旧TF_ON NO_GO、旧FAILED/retention及22收据缺口全部保留。
'''
    write('PF2-QUALITY-AND-COST.md', quality)
    route = '''# 路由、计数与状态审计

OBSERVED：同一wrapper、同一冻结TF_ON、相同reset bank；C全S，P从首chunk S开始按S,S,T重复。reset/cache RPC不计周期，S全拒绝仍计S，只有完整policy成功返回才推进。所有完整policy实际提交50个合法scheduler更新。

十二局的COUNTER/AUDIT/FINISH_READY齐全，独立报告组件均PASS_COMPLETE_EPISODE_COMPONENTS，无证据缺口；六对物理/controller/实际Python、NumPy初始RNG、init state、model XML、双相机保存原件逐字节一致。首次分叉前两个S的RAM有效KV/RNG/张量比较每对REFERENCE_STORED/PASS_EXACT。

必须保留例外：reset/S0_before的六个lazy scheduler字段（video/action的sigmas、timesteps、training）只比较首次消费前的有效输入，原始全字段差异如实保存；native infer会无条件重建20/50时间表。S0/S1完成后比较全部捕获字段，不把例外扩展到其他状态。完整KV仅现场RAM比较，持久化摘要不是离线全KV重验；未保存的模拟器内部字段UNKNOWN。

S保留原O1/O2/O3、K2/B2、bonus、prefix_then_cost和50更新覆盖。T为canonical CFG2 action full50加原末尾工作，51次B2 action forward；video/cache/final工作分别记typed_forward_calls。没有删旧ObservedAPI断言或伪造O3事件。

OBSERVED：前两个S的6个描述单位中，风险/对照的接受深度接近。风险最大cost/tau分别.8473/.9372/.8967，对照.9543/.9907/.9955；同任务平均score差值方向不一致，不能据此建立“风险越大越贴门限”的解释。完整分析见../pre-treatment-score-result-v1/pre-treatment-score-analysis.json。

UNKNOWN：这不排除后续chunk局部敏感性，不证明现有score能预测任务风险，也不重新计算未保存的teacher velocity范数。不能反向利用六点拟合新阈值或按失败task硬编码回退。

RECOMMENDATION：保留原生产接受规则与四false；新诊断应明确区分teacher数值路径、接受draft和机器人闭环，不把PF2当bonus、环境回滚或质量证书。
'''
    write('PF2-ROUTE-AND-COUNTER-AUDIT.md', route)
    write('CLAIM-EVIDENCE-MATRIX.md', '''# 论文主张与证据

|主张|本轮证据|允许写入的边界/缺口|
|---|---|---|
|PF2在此风险DEV救回部分失败|1/29、3/26本轮C败/P胜，初态与首两S审计齐|可写这两条观察，不外推总体成功率|
|当前PF2可作为最终TF配方|风险4/26仍失败；控制1/36长度1.5577|不允许，登记投入NO_GO|
|无控制组质量代价|控制3/3均成功，但罚分722→865、长度尾部超限|不允许用成功率隐藏轨迹代价|
|投机单独带来E2E加速|本轮没有原Target臂或正式独占计时|UNKNOWN，不以target calls或更早失败代替速度|
|PF2降低单位合法更新成本|探索性0.04068→0.05251秒；连续可比UNKNOWN|不支持下降；只列观察与限制|
|TF Transformer优于MLP|本轮冻结同一TF，两种调度；没有MLP臂|不允许架构优越性主张|
|teacher差异或累积误差为失败根因|PF2改变多个路径比例，前两个S无清晰风险分离|HYPOTHESIS，仍需独立机制诊断|
|reset优化使实验可更快重复|完整512-token BF16 bank实际全臂启用，12次15–107ms|可报实际reset时间；旧等价证据保留，不冒充MegaTransformer新贡献|
|MegaTransformer新系统协同收益|本轮无新kernel或shape×engine实验|本轮不新增支持，原系统正负结果另列|
|质量/非劣/TEST/正式性能放行|原四false和旧NO_GO保持|不允许写PASS|
''')
    life = resource['provisioned_server_lifecycle_hours']
    write('RESOURCE-AND-STOP-REPORT.md', f'''# 资源、终态与止损

OBSERVED：r3终态COMPLETE_RECORDED_SIX_PAIR_PANEL，十二完整机器人任务，runner生命周期1253.530秒（约20.89分钟）；从原授权11:11:41到r3终态累计7063.751秒（约1.962小时），含准备/等待/失败，未重置起点。并非从首次GPU active才计费。

三次已登记runner生命周期分别24.193/37.641/1253.530秒。r1 WAM配置模块导入冲突，在模型构造前失败；r2两client在环境初始化/transport前因LIBERO包路径失败，两个零环境步基础设施attempt，不能算算法失败或偷偷删除。修复source namespace后r3运行原固定面板，不按机器人结果挑重跑。旧r1/r2原FAILED/STOP/源码快照/报告均保留。

六个已申请server进程的登记到reap累计{life:.6f} GPU-worker生命周期小时（含两卡、初始化和失败准备；不是CUDA active时间）；实际全模型GPU active时间UNKNOWN。各CPU server2/client2，实际绑定0–7，分析单核12；没有draft DDP。

已计数完整episode target forward共16054，最大单局2676<4000。r2缺失COUNTER按原规则额外保守扣8000，未覆盖非episode调用另留1000；保守总账25054/49000、余额23946，不把UNKNOWN当0。原12局已完成，不因有余额扩任务。详见RESOURCE-DETAILS.json。

原23:11:41清理硬截止和22:56:41停新工作未变；实际13:09:25已清理作业。原始证据1267文件/55035793字节（约52.5MiB），远小于3GiB；迁移target权重24,359,646,412字节属于已批准准备资产，与新增研究证据分列。临时249MiB传输包已双端清理；本轮20.1MB证据传输包在完整回传/解包后可删除，原件保留。

两卡全部own进程组EMPTY；guard最终CLEAN，无遗留pins；后续真实资源观察85个登记身份无存活，nvidia-smi计算进程空、每卡1MiB/0%利用率。不是通过杀其他进程获得。实例关机NOT_REQUESTED_NOT_EXECUTED，租卡可能继续计费。

默认reset全臂使用同512-token BF16 bank，十二次耗时{min(x['sec'] for x in reset):.6f}–{max(x['sec'] for x in reset):.6f}秒；加载模型/首次bank构建单列。原阈值、旧FAILED、tile及491保护metadata不改。远端未运行smoke、独立测试或项目SHA；来源/大小/raw字节证明不冒充新的密码学复验。

止损：本轮投入NO_GO；不自动扩规模、补旧19/24项、训练、调tau、PF3/4、TEST或正式性能。时间余额不是追加实验授权。RECOMMENDATION：将下一个有界教师路径诊断交由新登记决定；10/22 go/no-go、10/23接口与claim冻结、10/28主表、11/01 18:00中文证据包的内部节点保留，证据不足缩主张。
''')
    write('PAIRED-DEV-EPISODES.csv', (RAW / 'report-v1/panel.csv').read_text())
    write('FACTS.md', '''# 本轮事实与导航

OBSERVED：同一TF_ON连续/PF2六对完整，C3/6、P5/6；救回2、丢失成功0。task4/26两臂都失败；控制1/36长度超限。登记投入NO_GO，四权限false，旧NO_GO/FAILED/retention保留。

UNKNOWN：总体LIBERO质量、独立非劣、相对原Target的正式单chunk/任务/E2E加速与全模型CUDA活跃时间。本轮不是架构选择或独立TEST。

HYPOTHESIS：周期完整target有部分风险改善信号，但teacher路径变化与draft提交频率共同变化，根因未分离。

RECOMMENDATION：暂冻TF64结构/API研究，不将当前权重+连续或PF2配方放入全量验收。新CPU分析不支持按首两S既有score直接识别风险；先做另行登记的数值/接受路径诊断，不凭猜想训练。

本目录五份主题报告及CSV是派生摘要；原始COUNTER/AUDIT/终态/命令、r1/r2失败、固定投入门和CPU重算均在上级job及remote-archive，未经改写。原始report-v1包含独立FACTS/QUALITY/SYSTEM/STOP和完整RESULT.json。
''')
    assert sum(p.stat().st_size for p in OUT.iterdir()) < 8 * 2**20
    write('CPU-FINAL.json', dict(status='COMPLETE_CPU_EVIDENCE_SYNTHESIS', elapsed_sec=time.monotonic() - started,
        pilot_decision='NO_GO', GPU=False, permissions=result['permissions']))
    signal.alarm(0)
    print(json.dumps(dict(status='COMPLETE_CPU_EVIDENCE_SYNTHESIS', out=str(OUT),
        provisioned_server_lifecycle_hours=life, reset_range_sec=[min(x['sec'] for x in reset), max(x['sec'] for x in reset)])))


if __name__ == '__main__':
    main()
