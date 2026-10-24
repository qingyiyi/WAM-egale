# 当前FFN交接与ACT驻留结果

本轮已实际完成四方案；三KV有限逐位/状态均通过，但局部FFN及正常整层均无收益，不采用。原采用A SELF-QKV-TMA-4MMA源码/DSO不变。不能把本轮负结果写成正收益。

固定review入口：

- [最终结果](../.pipeline/experiments/rtx5090-block-port-20261023/ffn-local-handoff-20261024/REVIEW_RETURN.md)
- [当前A完整阶段表](../.pipeline/experiments/rtx5090-block-port-20261023/ffn-local-handoff-20261024/STAGE-TABLE.md)
- [新候选源和结果入口](../.pipeline/experiments/rtx5090-block-port-20261023/ffn-local-handoff-20261024/SOURCE-AND-RESULT-ENTRY.json)
- [机器可读结果](../.pipeline/experiments/rtx5090-block-port-20261023/ffn-local-handoff-20261024/RESULTS.json)
- [主张与缺口](../.pipeline/experiments/rtx5090-block-port-20261023/ffn-local-handoff-20261024/CLAIM-EVIDENCE-MATRIX.md)

B仅专职输入、C跨任务交接，整层约增0.6%–1.2%；D M16/64KiB ACT驻留整层慢2.03–2.26倍、E cluster2/DSM双CTA慢2.75–3.12倍。逻辑权重tile请求增大不是HBM实测。5090实际双CTA DSM可运行，不能由慢结果断言硬件不支持，也不能将可运行写成加速。

全部eager/no Graph，强Torch/Guarded和独立A均保留。四科学门及production_admitted=false，单层重放不是机器人E2E。33本轮身份/阶段组退出，5090 context清零；双A100持续原RoboTwin固定队列，17:39为335/600有效组三臂。

公开省略：凭据、SSH部署/生命周期runner、权重、tensor/KV fixtures、DSO和工具链；源码、patch、原始标量计时/数值摘要、失败及资源收据保留。A源与各候选源分别固定在本轮namespace；早期prepare脚本不是最终源码入口。
