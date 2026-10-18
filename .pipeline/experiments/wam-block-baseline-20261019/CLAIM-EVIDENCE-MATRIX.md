# 主张与证据

| 主张 | 所需证据 | 当前证据 | 缺口 | 允许写法 |
|---|---|---|---|---|
| 当前独立实现只接管SDPA | 固定integration/install与CUDA launch | 已有源码与有限R1证据 | 其余子层仍Torch | 允许限定描述 |
| B0已显式执行完整block | 自身阶段传递、真实参数、完整输出/状态 | 六真实CPU fixture+六合法写控制通过 | GPU/其余层/原layout未验 | 仅CPU保存后契约 |
| 逻辑M由32变224 | 真实Linear输入与可flatten stride | CPU模块观测支持 | 实际CUDA GEMM/kernel未profile | 允许称逻辑几何 |
| candidate为独立Q16 | mapping/history/private代码与shape | CFG2×7、无跨候选attention | 非闭环质量证明 | 允许接口事实 |
| B0 GPU与R0逐位一致 | 六GPU输出+有效状态/RNG | NOT_RUN | 新授权/执行 | 不允许 |
| B0完整block性能接近原版 | 同cache A/A和三轮配对 | NOT_RUN；旧R1波动大 | GPU数值/可比较计时 | 不允许 |
| 静态投影缓存可复用 | 完整原形状、不可变条件、独立数值门 | 两个fixture CPU控制通过 | GPU/生产失效协议 | 仅静态CPU诊断 |
| MegaTransformer调度有增量 | R_strong/P_seq/P_sched、真实profile | NOT_IMPLEMENTED | D0和调度/因果数据 | 不允许 |
| 减少target调用带来机器人加速 | ΣT_round/Σg与完整质量/E2E | 本轮是block CPU回放 | 有效提交/闭环独立证据 | 不允许 |

四项quality/TEST/非劣/正式性能权限仍false；不继承其他模型/权重或旧版本的证书。
