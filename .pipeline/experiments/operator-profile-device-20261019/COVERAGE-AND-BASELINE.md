# 当前源码/路径覆盖

|路径|活动源码|实际覆盖|尚不能继承|
|---|---|---|---|
|R0 原Torch|投机采样/wam-p0-worktree/wan_va/modules/model.py|M26六block0原始GPU gold逐位；源字节等于ca5f37e|全部生产输入/未知原GPU布局|
|B0 本地顺序block|算子研究/wam_ops/baseline|forward AST同R0；M25两独立进程完整block/30层有限验证|M25严格性能全点门未过|
|R1 默认独立CUDA SDPA|算子研究/wam_ops/optimized/block.py和attention.py|M25有限30层/状态过；本轮未改默认入口|与M26新FFN组件的联合30层未测|
|R_strong 静态block|optimized/prepared.py，已有native/guarded，原形状条件缓存与Graph|M26 30eager/24Graph配置，六个实际block0|O1/O2/O3属于外层verifier，不能称本轮完整verifier组合证明；无生产cache失效协议|
|最终新组件|optimized/ffn.py及csrc/ffn.cu的ffn1_gelu_out|接入R_strong六真实block×两个进程；严格输出/状态过，原Torch FFN2保留|默认optimized未自动启用；未测30层/chunk/episode/接受路径|
|完整device FFN诊断|csrc/ffn.cu的ffn_out；首版源归档DEVICE-V1-FFN-SOURCE.cu|六点FFN1/GELU过、FFN2失败；保持NO-GO|不因收缩组件通过而改此失败|

R_strong是首轮已准入配置中七样本诊断后冻结的逐点参考；未声称任何时段全局最优。两独立进程的新/旧比较在相同冻结路径内交错，绝对延迟漂移保留。

组合逐项见COMBINATION-MATRIX.csv和原RESULT。cached/Graph准入只覆盖静态真实tensor；不会借旧测试或名称组合宣称生产证明。
