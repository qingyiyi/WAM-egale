# WAM 短状态

更新：2026-10-25T16:38:36.211404+08:00；动态作业实查：2026-10-25T16:33:47.595962+08:00。[当前计划](cycles/wam-research-20261016/PLAN.md)、[完整当前进度](cycles/wam-research-20261016/PROGRESS.md)、[现行约束](memory/CURRENT-CONSTRAINTS.md)。

## 运行与截止

- 本机M42：resume05恢复Goal/VAE资产，守护存活；校验→路径门→模型加载→固定完整DEV自动接续。新T0 15:56:53，硬截止21:56:53；23剩余case/90臂，单worker、CPU2+2、每局1800秒/清理600秒。旧15:45窗口、VAE路径/self-kill失败及0teacher准备记录保持。[实际status](experiments/suite-acceptance-calibration-20261025/multi-init-20261025/resume05/SUPERVISOR-STATUS.json)。
- 双A100 RoboTwin：556/600完整组、1668/1800局，117.111/160GPU小时，两guard存活。原281/406缺口与4未配齐完整记录单列，不自动重试。[实查](maintenance/hw-compact-20261025/RESOURCE-SNAPSHOT.json)。
- 5090：算子测量已闭合，当前带读，无新GPU候选。其他Cycle不在本轮整理范围。

## 当前结论与阻塞

M42原18attempt有10完整均成功、7严格初态无效case、1用户中断，0合格完整四臂组，不能选阈值。五套83030行CAL来自CONT主池，非当前ON的on-policy数据；按已登记OLD/Q75/Q90统一比较。质量/长度/接受深度为主，耗时附带，PFoff/full512 BF16 reset不变。

M39实际采用A170（SELF-QKV-TMA-4MMA）；历史三KV原Torch/A约1.25/1.76/2.18倍，限单层replay。C2比A慢12.27%–15.46%，G2绕过peer DSM仍慢8.71%–10.42%；C4、DSM8/R1也NO_GO。有限27tensor/七段/有效KV/RNG通过，不代表全部输入/闭环。源与成本支持新增交接/主循环/提交进度开销，纯DSM指令/HBM根因UNKNOWN。[最新全17阶段表与结果](experiments/sm120-ffn-next-20261024/CLUSTER2-ATTRIBUTION-20261025/REVIEW_RETURN.md)。

FFN同步合并仍为[CPU源码候选](experiments/sm120-ffn-next-20261024/FFN-SYNC-SOURCE-REVIEW-20261025.md)，A未改。每段FP32部分和加前一BF16运行和，末段加bias后舍入；不能增加不存在的独立partial舍入。

M41五环境44组三臂/132主表局已收口，TF/PF2整体质量未放行。Long任务E2E约1.121/1.062倍；Goal连续TF与Robo两投机臂存在减速，长尾/新增失败保留。Robo第二S在刷新前分叉，隔离FAILED；Long只有8共同组。[完整分母/时间账本](experiments/rtx5090-spec-speed-20261025/PERFORMANCE-RESULT.md)。

## 放行与下一步

`engineering_quality_screen=false`；`independent_test_complete=false`；`noninferiority_established=false`；`formal_performance_permission=false`。原M17配对/FIT-DEV缺口未解决，不启动TEST/正式45chunk或无限训练。

先跟进Goal完整DEV及Robo队列，再做FFN下一节带读/有界验证；10/28主表与11/01 18:00北京时间中文证据包是规划目标，旧10/22–23节点不补称完成。中文逐段写作[Prompt](prompts/WAM-PAPER-COLLABORATION.md)已交付。

H02核验归档后释放120.619GiB；原tile/源码/必要失败保留，临时目录`/rjs/cxz/tmp`。[旧全文与本次精简](maintenance/hw-compact-20261025/PLAN.md)。开新任务重读实际登记/status；此页不授予额外预算。
