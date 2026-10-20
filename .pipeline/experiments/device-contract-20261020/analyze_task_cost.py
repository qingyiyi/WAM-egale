"""Reproducible ratio-of-sums decomposition of existing complete M35 attempts."""
import csv
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
OUT=Path(__file__).resolve().parent
SOURCE=ROOT/'.pipeline/experiments/libero-four-suite-performance-20261019/ISOLATED-EPISODES.csv'
FIELDS=('episode_sec','steps','policy_chunks','policy_server_seconds','action_seconds',
        'policy_RPC_seconds','cache_RPC_seconds','reset_RPC_seconds','commits')

def totals(rows):
    d={k:sum(float(r[k]) for r in rows) for k in FIELDS}
    d.update(N=len(rows),successes=sum(r['success']=='True' for r in rows))
    d['non_RPC_residual']=d['episode_sec']-d['policy_RPC_seconds']-d['cache_RPC_seconds']-d['reset_RPC_seconds']
    d['policy_RPC_minus_native']=d['policy_RPC_seconds']-d['policy_server_seconds']
    d['policy_per_chunk']=d['policy_server_seconds']/d['policy_chunks']
    d['action_per_update']=d['action_seconds']/d['commits']
    d['mixed_nonaction_per_chunk']=(d['policy_server_seconds']-d['action_seconds'])/d['policy_chunks']
    d['seconds_per_success_including_failed_attempts']=(d['episode_sec']/d['successes']
                                                      if d['successes'] else None)
    return d

def main():
    rows=[r for r in csv.DictReader(SOURCE.open()) if r['complete']=='True']
    report=dict(source=str(SOURCE.relative_to(ROOT)),source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
                estimator='ratio of total times; same suite/repetition/case matched pairs',suites={})
    table=[]
    for suite in ('long','object','goal','spatial'):
        rr=[r for r in rows if r['suite']==suite]
        arms={a:[r for r in rr if r['arm']==a] for a in 'TCP'}
        sums={a:totals(r) for a,r in arms.items()}
        by={(r['repetition'],r['case_id'],r['arm']):r for r in rr}
        pair=[];bad=[]
        for t in arms['T']:
            c=by[(t['repetition'],t['case_id'],'C')]
            if t['success']=='True' and c['success']=='True':pair.append((t,c))
            if t['success']!=c['success']:
                bad.append(dict(repetition=t['repetition'],task=t['task'],init=t['init'],
                                T={k:t[k] for k in ('success','steps','episode_sec','policy_chunks')},
                                C={k:c[k] for k in ('success','steps','episode_sec','policy_chunks')}))
        common={a:totals([p[i] for p in pair]) for i,a in enumerate('TC')}
        t,c=sums['T'],sums['C']
        effect=dict(all_attempt_warm_speedup=t['episode_sec']/c['episode_sec'],
            common_success_warm_speedup=common['T']['episode_sec']/common['C']['episode_sec'],
            common_success_N=len(pair),per_policy_chunk_speedup=t['policy_per_chunk']/c['policy_per_chunk'],
            per_valid_update_speedup=t['action_per_update']/c['action_per_update'],
            C_to_T_chunk_ratio=c['policy_chunks']/t['policy_chunks'],
            C_to_T_actual_steps_ratio=c['steps']/t['steps'],
            T_action_fraction_of_native_policy=t['action_seconds']/t['policy_server_seconds'])
        effect['per_success_cost_speedup_including_failures']=(
            t['seconds_per_success_including_failed_attempts']/c['seconds_per_success_including_failed_attempts']
            if t['successes'] and c['successes'] else None)
        report['suites'][suite]=dict(arms=sums,common_success=common,effects=effect,discordant=bad)
        table.append(f"|{suite}|{t['successes']}/{t['N']} : {c['successes']}/{c['N']}|{effect['per_valid_update_speedup']:.3f}×|{effect['per_policy_chunk_speedup']:.3f}×|{effect['C_to_T_chunk_ratio']:.4f}|{effect['all_attempt_warm_speedup']:.3f}×|{effect['common_success_warm_speedup']:.3f}× ({len(pair)}对)|")
    (OUT/'TASK-COST-DIAGNOSIS.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    (OUT/'TASK-COST-DIAGNOSIS.md').write_text('''# TF任务加速比的可观测成本来源

OBSERVED：复用M35的330完整计时局，固定10任务/init49，独立server重复；不是原6000质量面板，不增加独立质量样本数。Object只有2重复，其他3重复。所有臂相同优化reset；本轮没有使用M34未准入整层kernel。

|套件|T : TF成功数|action每有效更新|完整policy每chunk|TF/T chunk数|全部尝试warm任务|共同成功warm任务|
|---|---|---|---|---|---|---|
'''+ '\n'.join(table)+'''

## Long：一个失败case的三次重复显著改变总任务工作量

三次重复的task4/init49均Target成功239命令/16chunk，TF失败796命令/50chunk。它们是同一case的重复，不能称为三个独立风险初态。T合计225.55秒，TF545.57秒；其余27个共同成功尝试T2145.92秒、TF1597.00秒。全30次TF命令数9363对7862（+19.09%），chunk602对515（+16.89%）。因此每chunk1.363×仍只得到全尝试1.107×；共同成功1.344×只作为条件指标，不能替换原全样本。这里识别的是成本来源，不是该case失败的算法根因；后者UNKNOWN。

## Object：action之外约三成policy工作、额外chunk及推理外成本

20次双方全成功。action每有效更新1.575×；Target平均每chunk3.820秒中action2.671秒、混合剩余1.149秒。TF对应2.848/1.696/1.152秒，剩余几乎未减少，因此完整policy每chunk1.341×。该剩余是计时边界之差，不能直接全称为video或VAE。

TF184chunk对T178（+3.37%），实际命令2746对2634（+4.25%）。原生policy合计从680.01降到524.12秒，但完整warm任务还包含policy包装/RPC差、cache RPC、reset及非RPC剩余。TF各项分别6.72/44.23/0.712/96.27秒，合计147.93秒；全任务最终817.32/672.04=1.216×。非RPC剩余包含环境与客户端工作，缺细分日志，不能全部归为env.step或CPU争用。reset仅TF全任务约0.106%，不是当前优先瓶颈。

## 套件间比较与PF2

Goal有一次Target失败/TF成功，Spatial有相互胜负；全尝试倍数较高同时受到轨迹工作量变化影响。共同成功倍数Goal1.319×、Spatial1.253×，与Object的差距比全部尝试口径小。不同套件不能仅用全部尝试倍数判断GPU实现优劣。

PF2增加完整Target chunk；Long平均policy从TF2.905秒到PF2的3.249秒，Object2.848到3.142秒，约增加11.8%/10.3%。这是实际policy成本，不由forward数代替。是否改善质量须另看完整结果，不能因为有refresh就假设状态恢复。

RECOMMENDATION：计算效率以固定工作量、共同成功的同case总时间比为条件性指标，同时并列SR、全attempt成本与含失败成本的每成功任务耗时。保留慢成功和失败；不得各臂单独挑成功集合。完整任务质量/状态门未通过，当前为工程成本观察，formal_performance_permission=false。warm边界不含模型冷装载、env.close和落盘；首两S审计开销仍在客户墙钟中。

Long全尝试时间除以成功数：Target 2371.47/30=79.05秒，TF 2142.57/27=79.35秒，约0.996×。这解释为何不能只用共同成功1.344×描述实际成功产出成本。这仍是本次固定init49工程面板，不能替代原各500局成功率或推断总体。

对于“以后是否只比较成功平均加速比”的问题：当前建议先报告同case共同成功的ratio-of-sums，三臂直接横比时使用三方共同成功集合或清楚列出各自配对N；同时必须列全attempt/SR/失败罚分和含失败成本的每成功任务耗时。这是建议，不把问句当作用户已采纳的长期政策。不能平均各臂不同成功集合，不能排除慢成功。名义同case配对仍受原首两S审计FAILED限制，未宣称严格状态等价。

UNKNOWN：未持久化video/观测编码/VAE各自独立跨度，未定位Long失败机制；不能从上述残差直接推出某个子阶段或硬件争用根因。

证据：[原330局CSV](../libero-four-suite-performance-20261019/ISOLATED-EPISODES.csv)、[原成本账本](../libero-four-suite-performance-20261019/COST-LEDGER.csv)、[时间边界](../libero-four-suite-performance-20261019/TIME-BOUNDARY.md)、[逐项机器结果](TASK-COST-DIAGNOSIS.json)。
''')

if __name__=='__main__':main()
