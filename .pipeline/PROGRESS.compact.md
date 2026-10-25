# WAM 短状态

更新：2026-10-25T18:05:45+08:00；动态作业实查：2026-10-25T17:59:00+08:00。[当前计划](cycles/wam-research-20261016/PLAN.md)、[完整当前进度](cycles/wam-research-20261016/PROGRESS.md)、[现行约束](memory/CURRENT-CONSTRAINTS.md)。

## 运行与截止

- 本机M42：resume05已闭合；81attempt、54完整均成功、4合格完整四臂组，其余身份/初态/基础设施排除保留。不平衡暴露DEV，尚不能选最终阈值。进程清理完成，原T0/21:56截止不改。[本轮结果](experiments/suite-acceptance-calibration-20261025/multi-init-20261025/resume05/RESULT.md)。
- 双A100 RoboTwin：569/600完整组、1707/1800局，119.952/160GPU小时，两guard存活。原281/406缺口与未配齐记录单列，不自动重试。[17:59实查](experiments/sm120-ffn-next-20261024/FFN-STAGE-SPLIT-20261025/ROBOTWIN-PROGRESS.json)。
- 5090：FFN两阶段同步版S已测完并采用，17:59清理。2865调用；原A独立保留。

## 当前结论与阻塞

M42原18attempt有10完整均成功、7严格初态无效case、1用户中断，原0合格四臂组属于更早attempt；resume05现有4组，仍不能直接定最终阈值。五套83030行CAL来自CONT主池，非当前ON的on-policy数据；按已登记OLD/Q75/Q90统一比较。质量/长度/接受深度为主，耗时附带，PFoff/full512 BF16 reset不变。

M39当前采用S（FFN两阶段同步版）：FFN1+GELU写global ACT后grid同步，再原FFN2七段BF16递推与残差；tile/warp不变。三KV/两独立进程、27tensor/七段/有效KV/输入/RNG和公开入口均有限逐位通过；完整层原Torch/S约1.288/1.797/2.238倍，相对A延迟下降2.25%/1.82%/1.90%。不是机器人E2E。[18阶段表/结果](experiments/sm120-ffn-next-20261024/FFN-STAGE-SPLIT-20261025/REVIEW_RETURN.md)。

FFN1+GELU约0.80×Torch，FFN2+残差约1.03×；下一步先分解FFN1主体/尾块的GEMM与后处理成本，再决定tile优化。C2/G2/C4/DSM8历史NO_GO保留；没有新的DSM/HBM硬件因果证据。

M41五环境44组三臂/132主表局已收口，TF/PF2整体质量未放行。Long任务E2E约1.121/1.062倍；Goal连续TF与Robo两投机臂存在减速，长尾/新增失败保留。Robo第二S在刷新前分叉，隔离FAILED；Long只有8共同组。[完整分母/时间账本](experiments/rtx5090-spec-speed-20261025/PERFORMANCE-RESULT.md)。

## 放行与下一步

`engineering_quality_screen=false`；`independent_test_complete=false`；`noninferiority_established=false`；`formal_performance_permission=false`。原M17配对/FIT-DEV缺口未解决，不启动TEST/正式45chunk或无限训练。

先审计Goal四组及Robo队列，再定位FFN1分项成本；10/28主表与11/01 18:00北京时间中文证据包是规划目标，旧10/22–23节点不补称完成。中文逐段写作[Prompt](prompts/WAM-PAPER-COLLABORATION.md)已交付。

H02核验归档后释放120.619GiB；原tile/源码/必要失败保留，临时目录`/rjs/cxz/tmp`。[旧全文与本次精简](maintenance/hw-compact-20261025/PLAN.md)。开新任务重读实际登记/status；此页不授予额外预算。
