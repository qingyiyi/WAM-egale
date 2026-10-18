# 固定Torch算子对齐：去除fast_math后的有限实证

结论：独立v2在本次全部真实数值与状态输入中恢复逐位一致，**新二进制性能尚未复测**。旧standalone完整30层失败原样保留，不能改写其身份。

| 层级 | 新v2结果 | 边界 |
|---|---|---|
| 9保存输入 | 9/9逐位，输入字节/布局/device/RNG未改 | 同样输入旧版本3/9逐位 |
| 真实attention | 108/108逐位 | 54点×self/cross，B2 video/action及B14 action |
| 完整block | 54/54逐位 | 三KV×first/last×层0/8/29，全部有效KV/metadata不改 |
| 完整30层verifier | 6/6 raw/CFG/有效动作/输入/全部有效KV/RNG严格通过 | 7候选各Q16、CFG2；旧两种注意力均替换，非shared-only |
| 完整canonical policy | 短/中/长3/3配对严格通过 | video20/action50，Torch和新版本各72次model调用；真实冻结重放，非机器人任务 |
| CPU独立保存输出复核 | 27/27一致、未初始化CUDA | 24完整模型raw输出+3对完整policy返回；全KV只做现场比较，不复制持久化 |

## 实际修正与可用入口

新目录：`投机采样/experiments/operator_torch_alignment_v2/`。固定Torch2.9.0+cu126及对应FlashAttention/CUTLASS，179个vendor文件逐字相同；CUDA入口去命名空间差别后逐字相同。数学编译唯一变化是去除`--use_fast_math`。命名空间与扩展名隔离，不调用Torch SDPA fallback。来源、许可证与固定官方CMake/flash_api已保存。

这构成对**该编译因素在本次失败输入上影响结果**的受控证据；尚未定位它内部FTZ/除法等具体指令，不猜哪一种是唯一微观根因。不能由有限测试证明所有BF16输入/其他编译器/其他GPU完全逐位。

`integration.py::install/restore/installed`实例级替换模型30层self/cross共60个attention op；真实完整模型和policy验证实际调用install/restore。`server_entry.py`提供独立canonical服务入口，未更改原模型源码或默认配置。该RPC服务入口本轮未另启动，不能声称已跑新机器人成功率。原tile保留。

## 状态与输入边界

真实input来自M21相同固定来源。action K288/1152/2160，video K272/1136/2144；H24/D128/BF16。三完整policy分别从相同的声明mutable状态、初态输入、Python/NumPy/CPU/CUDA RNG开始，比较完整返回动作、字段、30层全部有效KV/metadata、streaming VAE、video/action scheduler与末态RNG。两臂各72个真实forward，包含21video与51action调用的canonical终端工作；更新数与调用数不是同义词。

这是三组冻结policy请求，未执行LIBERO环境；未验证完整投机接受器/bonus/路径到闭环成功率，不签整项目质量门。所有四false、TF旧NO_GO、默认A_off保持。

## 性能还缺什么

M21 .9812/.9817等时间属于旧fast_math二进制，不能转借给新v2。新`worker_performance.py`和PERFORMANCE-RETEST-PLAN已准备；尚未启动。收尾时本机存在lhw GPU PID2015053，约1090MiB，不能作为独占正式计时窗口。后续需同真实输入、同计时边界重测kernel/完整block，再完整30层/完整policy；保留全部慢点、layout/pack和强B14基线，不能只报最好点。

## 资源与终态

CPU编译CUDA不可见、单ninja worker/两核，约十分钟；未安装系统依赖。r1九输入生命周期22.950秒；r2完整验证生命周期592.107秒，worker完整714次target forward，峰值21.847GiB。累计615.057/1200秒，剩余584.943秒，CUDA active未知。新增源/编译/证据约40.71MiB≤512MiB，未复制大权重或完整KV。

两作业returncode0，所有已登记自有PID按出生时间/UID核实退出。原main收尾发出未显式destroy_process_group的NCCL warning，原日志保留；不能因returncode0忽略清理，已实际核对无该进程与自有GPU context。未向其他用户发送信号。一次只读状态查询碰到/proc正好退出导致FileNotFoundError，不是实验失败。

## 证据

RESULT.json、NINE-INPUT-RESULT.csv、ATTENTION.csv、BLOCK.csv、FULL30.csv、POLICY.csv、CPU-SAVED-OUTPUT-AUDIT.json、SOURCE-COMPARISON.json、BUILD-ONLY-CHANGE.patch、OWNED-CLEANUP.json。原始job：`/rjs/cxz/论文探索/投机采样/results/runtime/p0-operator-torch-alignment-v2-20261018-r1`与`/rjs/cxz/论文探索/投机采样/results/runtime/p0-operator-torch-alignment-v2-20261018-r2`；EXECUTED源码在各job，原张量和二进制不公开。
