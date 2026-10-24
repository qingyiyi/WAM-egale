# WAM短状态入口

当前Cycle：wam-research-20261016；M39单层算子，M40 RoboTwin。2026-10-24 17:39+08检查。

当前保留SELF-QKV-TMA-4MMA：原Torch完整层KV288/1152/2160为1.2514/1.7546/2.1902×；Guarded为1.1127/1.0845/1.0453×。三方eager、无Graph、同CrossKV缓存，单层真实重放，非在线verify/机器人E2E。

本轮已实际完成FFN四方案：B一TMA输入分工、C一TMA/三写回/四MMA跨任务交接、D 64KiB ACT单CTA驻留、E双CTA cluster/DSM共享。真实三KV27中间量/输出、输入/全部有效KV/RNG和各schema有限逐位通过，但四者两进程无插桩整层及最后同轮FFN区间均慢于A，不采用。B/C整层约增0.6%–1.2%，D2.03–2.26倍、E2.75–3.12倍A；M16驻留破坏权重行复用，逻辑tile请求增加，非HBM实测。5090实际cluster2/DSM可运行；不能将可用等同高效。D首次编译失败保留。当前A源码/DSO本地远端六文件byte不变。

本轮13380完整block，作业生命周期和1207.323秒，编译1095.558秒，GPU工作进程生命周期108.073秒、CUDA active未知；1编译失败、0timeout，33自有身份/阶段组退出，5090 context为空，实例仍开。用户允许持续使用现有5090，旧十小时收据/截止/失败保留，逐job仍有界；不扩大A100或租新机。

M40双A100继续固定RoboTwin600有效三臂组（1800局），17:39为335组/1005局，2执行、262未领，两worker运行。旧STATUS竞态/pair281 C中断保留，原尾部一次补齐与跨namespace审计；原160GPU小时/10天不变。完成条件有效覆盖，不是worker退出。

质量边界保持：TF64结构为研究载体，旧TF_ON连续配方NO-GO、旧72/96 PARTIAL及三风险失败保留；不拼旧T/DEV/新50为独立TEST。PF2不同于bonus，不恢复物理状态。engineering_quality_screen=false、independent_test_complete=false、noninferiority_established=false、formal_performance_permission=false、production_admitted=false。新算子未验证其他层/在线接受/完整模型/机器人质量。

下一步建议保留大M与共享全局ACT，单独定位FFN1写回/尾部及K片段ready后再选新候选；本轮未展开完整DAG/新模型。算子仅optimized，baseline/投机/原tile/protected不改。不运行smoke、不做独立SHA、不杀他人进程、不改驱动/服务/时钟；BAAI仅~/cxz写。论文中文由用户飞书整理，不新建论文草稿。11/01 18:00证据包内部节点，具体科学主张随实际证据收缩。

- [完整当前进度](cycles/wam-research-20261016/PROGRESS.md)
- [FFN结果/止损](experiments/rtx5090-block-port-20261023/ffn-local-handoff-20261024/REVIEW_RETURN.md)
- [当前完整阶段表](experiments/rtx5090-block-port-20261023/ffn-local-handoff-20261024/STAGE-TABLE.md)
- [固定源与结果入口](experiments/rtx5090-block-port-20261023/ffn-local-handoff-20261024/SOURCE-AND-RESULT-ENTRY.json)
- [原Self QKV采用](experiments/rtx5090-block-port-20261023/projection-warp-specialized/self-qkv-tma-migration/REVIEW_RETURN.md)
- [双卡持续接续](experiments/robotwin-pilot-20261020/eval100-20261022/resume-after-storage-race-20261024/PLAN.md)
