---
kind: progress
cycle: wam-research-20261016
updated: '2026-10-21T13:04:10.565362+08:00'
---
# 当前研究进度

| ID | 阶段 | 状态 | 证据/下一步 |
|---|---|---|---|
| M17 | LIBERO证据收口 | in_progress | 6000局齐；曝光/首两S/长尾缺口保留，非独立TEST |
| M39 | 简易单层先对齐 | in_progress | 当前嵌入差小于约1.4%；FFN2小幅改善，仍慢15%–30%，先FFN1再ready |
| M40 | RoboTwin迁移 | in_progress | CURRENT18/审计/ON/CAL全部完成；DEV90入口待审查、未排队 |

## M39 已完成本轮

[完整结果](experiments/single-layer-async-gemm-20261020/ffn-resource-control-20261021/RESULT.md) / [原因核验](experiments/single-layer-async-gemm-20261020/ffn-resource-control-20261021/CAUSE-ANALYSIS.md)。前轮Cross/FFN函数组织改动已缓解旧集成惩罚。本轮同源码、同callee、同前序、同调用点边界，独立/整层两进程三KV差−1.33%至+1.28%；不能将旧约590µs独立与新整层混比，也不能把全部差额归给全局同步。

唯一新候选：专用FFN2阶段，M128输出tile、48输出×2个K7168分段=96任务并行，保留首段BF16及第二段带bias的舍入/归并顺序。9真实签名、每格23中间量、18有效KV转移、输入/RNG/重复输出有限逐位通过，max_abs=0；一个kernel。不是任意输入或闭环证书。

| KV | native ms，两进程 | 当前 ms，两进程 | 本轮改善 | 仍慢 |
|---|---|---|---|---|
| 288 | 2.1166 / 2.1181 | 2.7469 / 2.7443 | 0.630% / 0.649% | 29.78% / 29.56% |
| 1152 | 2.4039 / 2.4008 | 3.0069 / 3.0066 | 0.517% / 0.581% | 25.08% / 25.23% |
| 2160 | 2.8431 / 2.8390 | 3.2615 / 3.2617 | 0.403% / 0.358% | 14.72% / 14.89% |

两独立进程、固定60预热/12交错；GPU0单worker，GPU1机器人并行，SM1410/MEM1215。uncached/eager完整单层，非强cached/Graph、30层、chunk或E2E。

选后同边界三KV诊断：Cross K约625µs、V约619µs、FFN1约229µs、FFN2约143µs。独立/整层差−1.144%至+0.674%，插桩扰动−0.124%至+0.134%。相同外侧marker的FFN1约232–234µs对native165–167µs，FFN2约146–147µs对native116–118µs：继续优化自有阶段本身。

静态255regs、98848B shared、1CTA/SM、1456B local不变；独立local少但耗时相近，不能据local字节断言动态spill开销。NCU动态occupancy/HBM/L2/spill UNKNOWN。旧generic linear并行K负结果保留；当前专用阶段不同实现，未恢复旧版本。仍按阶段grid.sync，跨阶段ready未实现。

本轮GPU81.559秒、CPU准备396.740秒，M39累计1852.227/36000秒；owned退出、GPU0释放。原截止/失败不改。下一步FFN1主循环/tile/资源；随后才按真实DAG补位，各阶段不凑108任务。

## M40 后台链自然完成

[终态与数据边界](experiments/robotwin-pilot-20261020/current18-followup-20261021/RESULT.md)。13:02北京：CURRENT18齐，13成功5失败，CONT3000生成FIT12/CAL6；不是ON闭环SR。5779 FIT/3502 CAL监督审计通过，ON固定500updates完成，11195行部署adapter/decode有限逐位、0接受翻转。没有重校准tau；teacher B2/B14一致与闭环质量尚未通过。

108完整512-token BF16 reset bank沿用；完整episode bank等价未另测。guard/worker均自然退出，未人为暂停。M40累计7.1672/24h；DEV90三臂入口/首两S与成本审计尚待核查，没有排队，不宣称迁移完成。旧watch等待字段17不覆盖最终18/AUDIT。

## 共同边界

LIBERO T/C/P各500：Long483/484/473，Goal463/464/467，Spatial472/466/464，Object497/498/499。非独立TEST；paused72、A_off、retention/FAILED及四质量/性能权限false不变。baseline/原tile/LIBERO生产与state/cycle/rules保护；BAAI只写~/cxz。GitHub只main小证据。10/22 go/no-go、10/23冻结、10/28主表、11/01 18:00证据包。

[前完整检查点](cycles/wam-research-20261016/history/ffn-resource-control-20261021/PROGRESS.md)。
