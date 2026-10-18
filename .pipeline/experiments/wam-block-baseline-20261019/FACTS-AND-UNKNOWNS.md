# 完整block基线：事实与未知

生成时间：2026-10-19T02:11:57+08:00（北京）。科研base：`ca5f37e0482c2e38e3738e3d8e71e3b185990a23`。本轮为CPU准备，不启动新GPU。

OBSERVED：已有v2只替换60个self/cross SDPA入口，投影/norm/FFN仍为原Torch。固定提交包含108 attention、54 block、六30层query、三短中长冻结policy的有限GPU数值证据；这不是全自有设备block或机器人闭环证书。

OBSERVED：现在已新增可单独调用的B0_HYBRID_SEQUENTIAL，真实参数与显式self→cross→FFN阶段；没有调用原block.forward代理完成B0。最终版本六个真实输入+六个同输入合法写控制，共36次CPU完整block执行，R0/R0自复现、R0/B0输出逐位、有效KV有限及状态/输入/RNG通过。重复核验不增加独立样本数。

| 真实fixture | update_cache | CPU R0/B0字节相同 | 有效状态通过 | CPU R0/旧GPU gold字节相同 | CPU/旧GPU max_abs |
|---|---:|---|---|---|---:|
| short-B14-action-layer0-first | 0 | True | True | False | 1.5 |
| middle-B14-action-layer0-first | 0 | True | True | False | 1.5 |
| long-B14-action-layer0-first | 0 | True | True | False | 1.5 |
| short-B2-action-layer0-first | 0 | True | True | False | 1.0 |
| middle-B2-action-layer0-first | 0 | True | True | False | 1.0 |
| long-B2-action-layer0-first | 0 | True | True | False | 1.0 |

最后两列保留不同设备数值环境的真实差异，不能拿CPU通过覆盖GPU待验。CPU结果主张只涉及本轮相同CPU环境的R0与B0。

OBSERVED：最终源的short-B2、long-B14各28个可观察公共边界通过，静态原形状context K/V投影的R0/B0控制均通过CPU。新的静态block适配器仅供固定输入诊断，不提供生产缓存失效协议。

OBSERVED：22项CPU契约/负例通过，包括逻辑字节与非有限数、每臂模块/cache独立且同原Parameter对象共享、无GPU授权拒绝、模拟Event的状态恢复及ABBA/BAAB。模拟Event只检验计时协议，不是延迟测量。完整GPU driver已准备，尚无GPU执行证书。

OBSERVED：读取固定提交21个小源与16份既有结果，均存在；本地存在的相应文件逐字一致。未做权重/tensor hash。参数为原checkpoint block0的27项，共163,656,704参数、327,313,408字节，只加载单block到CPU。

UNKNOWN：B0在GPU的数值、状态、延迟、峰值显存、CUDA实际GEMM dispatch、完整模型及policy集成；原生产CUDA stride/storage offset/alias及compact中缺失的原RNG、原始sigma/timestep数值（仅精确temb由原PT保留）；其余层/时间段、完整video block的新B0证书。不得补造。

HYPOTHESIS：较大的候选行数可能帮助后续设备调度，尚无本轮GPU因果证据。旧计时波动的CPU/host/设备根因未分离。

RECOMMENDATION：下一步唯一优先是新授权下的六fixture GPU完整block数值/状态→A/A→匹配配对计时；不是马上开发persistent scheduler。四项机器人质量/正式性能权限仍false。
