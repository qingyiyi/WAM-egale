# RTX5090当前算子：Self QKV TMA迁移已采用

本次为真实单层block研究，不是机器人E2E。请固定本文件所在commit阅读，不将旧实验终态或质量门改写。

- 当前实现：W0 TMA、W1–3 epilogue、W4–7 MMA用于Self Q/K/V、Self输出、Cross Q、Cross输出。QKV共288任务；单独投影96任务；tile64×128，保留9×K320+K192十段BF16边界。外部阶段同步保留，没有完整DAG调度。
- 原Torch/新版整层短中长1.2475/1.7553/2.1872×，相对冻结旧版延迟降2.942%/2.808%/2.249%；Guarded/新版1.1021/1.0850/1.0433×。全部eager/no Graph、同缓存，两独立进程交错计时。阶段及整层不同口径分列。
- 三真实B14/Q16/block0的27中间量/输出、有效KV/输入/RNG、debug/normal、公开默认/非默认流有限逐位通过；不外推其他层/在线接受/完整模型。四科学门和production_admitted均false。
- 新增现有5090使用授权已登记，旧预算收据保留。双A100继续固定RoboTwin，14:01为299/600有效组三臂，正在执行；不能将其当最终结果。

先读[结果](../.pipeline/experiments/rtx5090-block-port-20261023/projection-warp-specialized/self-qkv-tma-migration/REVIEW_RETURN.md)、[完整阶段表](../.pipeline/experiments/rtx5090-block-port-20261023/projection-warp-specialized/self-qkv-tma-migration/STAGE-TABLE.md)、[当前源码](../.pipeline/experiments/rtx5090-block-port-20261023/workspace/算子研究/wam_ops/optimized/csrc/whole_layer.cu)、[固定入口](../.pipeline/experiments/rtx5090-block-port-20261023/projection-warp-specialized/self-qkv-tma-migration/SELECTED.json)。原Torch/Guarded/旧A/新版本原始样本在本轮results；旧A/新源码均保存。

源码/结果公开，未上传权重、真实tensor/KV、二进制、SSH部署脚本、凭据或工具链。固定DSO及fixture只在私有原证据中，不能声称网页端独立重算已完成。当前workspace是5090路径；公开根目录算子研究下的旧实现仍为历史，不能据它替代本轮源码。

下一步优先对融合FFN和三个96tile小投影做有限归因；不把调度、TMA或更多warp视为必然增益。历史负结果：FFN TMA流水、三个小投影细粒度split-K、原8MMA连续TMA均保留，详见RTX5090目录各REVIEW_RETURN。
