---
kind: progress
cycle: wam-research-20261016
updated: '2026-10-21T21:15:38.031247+08:00'
plan: PLAN.md
current: M39
next: M39
---
# 当前研究进度

| ID | 阶段 | 状态 | 证据 | 下一步 |
| --- | --- | --- | --- | --- |
| `M17` | LIBERO证据收口 | in_progress | 6000局齐，非独立TEST | 曝光/首两S/长尾缺口保留 |
| `M39` | 简易单层先对齐 | in_progress | 小投影固定阶段整层改善1.88%–2.08%，Self-Q退化保留 | Self-Q编译边界，再FFN原语，最后DAG |
| `M40` | RoboTwin迁移 | waiting-review | 固定DEV90完成、30配对审计通过 | 小样本质量结论与扩大初态建议；不自动启动 |

## M39：六小投影部分改善

[本轮结果](../../experiments/single-layer-async-gemm-20261020/small-projection-function-20261021/RESULT.md)、[整层表](../../experiments/single-layer-async-gemm-20261020/small-projection-function-20261021/WHOLE-LAYER-COMPARISON.csv)、[旧新逐阶段](../../experiments/single-layer-async-gemm-20261020/small-projection-function-20261021/STAGE-BEFORE-AFTER.csv)。只改optimized的`action_projection`及B14路由，固定M224、N/K3072；64×128、96任务、两K1536分段/BF16边界/claim/sync均不变。源码e91e09eb、DSO23238b37两机一致；baseline/投机不动。

两进程三KV同轮Graph完整层：短约2.563→2.513ms，中2.777→2.719ms，长3.027→2.970ms，均改善1.88%–2.08%。9真实签名、23共有中间量、18有效KV转移及输入/RNG有限逐位，max_abs0；阶段240输出比较与6组状态检查通过。不是30层或机器人质量门，不与旧倍数相乘。

必须保留退化：五小投影整层内更快，Self-Q约51.2→57.3–58.4µs；新独立调用体也比旧独立慢，不能靠换独立参考宣称全对齐。整层localSize1464→1480B，尚无动态spill/等待分账。改函数组织有净收益，不足以把根因唯一归于spill。FFN1独立仍慢约34%–35%，FFN2约24%–28%；ready-DAG尚未实现。

同前序主表/整层边界1410MHz；三个hot-only context样本1395MHz原样保留。插桩扰动−0.244%至+0.017%。[SASS静态分析](../../experiments/single-layer-async-gemm-20261020/small-projection-function-20261021/SASS-STATIC-SUMMARY.json)不是动态性能计数；前轮bank地址/NCU吻合证据仍保留。

本轮GPU109.306358秒、CPU准备/编译272.149908秒，累计**2483.115753/36000秒**；0GPU失败/timeout，报告字段读错一次的CPU错误已修复且留账。仅一个候选，未扩展搜索。新22:23:55及旧所有截止保留；21:08 PID/出生身份确认owned全部退出、两GPU空闲。[资源/清理](../../experiments/single-layer-async-gemm-20261020/small-projection-function-20261021/RESOURCE-AND-STOP-REPORT.json)。

## M40：RoboTwin DEV90终态

六任务×每任务5初态（valid14–18）×T/C/P，**90/90**。每臂30局：Target25、TF连续24、TF+PF2 23；30组C/P初态及首两S通过。[完整结果](../../experiments/robotwin-pilot-20261020/dev90-20261021-r2/TERMINAL-RESULT.md)、[逐对与请求成本](../../experiments/robotwin-pilot-20261020/dev90-20261021-r2/TERMINAL-SUMMARY.json)。该小面板不支持保留Target成功率的宣称，非完整RoboTwin/独立TEST。

实际命令T/C/P为12224/13376/13408，任务horizon罚分12124/13256/13268；RoboTwin完整chunk可能越过horizon，实际与罚分分列，不套LIBERO800。双方成功长度比此面板均1.0，不能删除失败。成本为插桩探索性，不作正式E2E倍数。

GPU本段24199.408206秒，累计50006.379538/86400秒；原10/22 01:26:04.706686截止不变。guard结束、server清理−15、90客户端0，owned均退出；没有追加任务。用户询问5初态是否太少，当前建议扩到6×50×3=900局须重新前瞻分区和有界安排；仅建议，不作为执行授权。

## 共同边界

四权限false、paused72、A_off、旧retention/FAILED保持；baseline/原tile/投机生产/保护文件不动，BAAI仅~/cxz。LIBERO三臂各500：Long483/484/473、Goal463/464/467、Spatial472/466/464、Object497/498/499；每任务50初态，非独立TEST。10/22 go/no-go，10/23冻结，10/28表，11/01 18:00证据包。

[前完整记录](history/small-projection-function-before-20261021/PROGRESS.md)。
