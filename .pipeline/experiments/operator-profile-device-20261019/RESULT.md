# 强基线与FFN设备原语：本轮结果

状态：COMPLETE_BOUNDED_STEPS_WITH_PARTIAL_DEVICE_SUPPORT。强基线/profile完成；完整device FFN未过，收缩的FFN1+GELU在六个真实block、两个独立进程通过。baseline和1875份投机源码/配置未变；默认安装入口不自动启用新原语。

## OBSERVED：强基线与真实输入

- 复用六个保存的实际block0输入：B2/B14 × KV288/1152/2160，每候选仍Q16；实际hidden3072、FFN14336。只按safetensors索引读取block0，163656704参数；没有加载完整target或采新机器人轨迹。
- 30个eager配置和24个Graph配置全部通过输出/全部有效KV/所有metadata/输入/RNG门；原始GPU gold亦逐位相同。仅update_cache=0静态场景，本轮不新增write1/2、完整30层或policy等价证明。
- 原形状cross条件K/V投影冷缓存：B2 12MiB，B14 84MiB；不从B2复制成B14。native/guarded均为已有shared-prefix组件，固定guard短/长为G2、中为G1。两者全部保留对照，profile后冻结六点可用最强组合。
- Graph之前独立admit；Python guard不在replay中重跑。slot/条件/权重改变需要新准入。冷填充、准备与Graph构建不计replay；这是静态已准入边界，不是生产失效协议。
- B2 prepared路径直接按原排序读取临时KV，返回metadata与原版相同；未宣称转瞬mask写入过程或并发可观察性等价。B14各候选私有KV保持独立。

覆盖与逐组合字段见[COVERAGE-AND-BASELINE.md](COVERAGE-AND-BASELINE.md)和[54行eager/Graph矩阵](COMBINATION-MATRIX.csv)。R_strong为首轮七样本诊断后冻结的逐点参考，不声称所有时段全局最优；默认独立CUDA SDPA与新FFN组件的联合30层未测。

## OBSERVED：shape与profile

实际aten::addmm证实线性层把候选与token合并：M32→224。self Q/K/V/out及cross Q/out的N/K均3072/3072；FFN1 N/K14336/3072，FFN2 N/K3072/14336。未缓存时cross K/V为M1024/7168，缓存命中后这两次GEMM消失。attention每个实例的Q仍16，未构成可跨候选注意的新Q112序列。

完整block依赖：FP32 norm1/调制→BF16 QKV→Q/K RMSNorm→原FP64 RoPE→合法self attention→输出投影/门控残差→cross norm/attention/残差→FP32 norm3/调制→BF16 FFN1→BF16 GELU(tanh)→BF16 FFN2→FP32 gate/residual→BF16。FFN2每行归约需要14336维activation就绪。

Chrome trace只汇总cat=kernel；通过CUDA-runtime correlation映射到CPU模块范围。gpu_user_annotation是时间范围，不与kernel相加；原profiler raw聚合的重叠范围仅留原件，不作为占比。强cache路径FFN占实际trace kernel时长21.9%–29.9%，因此选择一个FFN子图。该占比来自带hooks的eager profile，不冒充Graph延迟或整个WAM占比。

## OBSERVED：最终最小原语与完整block

首版CUDA独立GEMM+GELU+GEMM：六点FFN1及独立GELU逐位过，FFN2六点均失败。B2差异39123–39247/98304元素；B14差异120892–125318/688128；最大绝对差0.25，均为隐藏激活单位，不能解读为动作误差。完整device FFN拒绝集成。

实测FFN2 gridZ为B2=7/B14=2；在两个真实short输入上核验FP32/BF16 workspace和两种serial BF16累加，共四个固定公式，均未逐位复现。它们没有证明split-K是根因；没有继续猜配置或修改阈值。

最终组件只覆盖FFN1+GELU：CUTLASS SM80 CTA64×128×64、warp32×64×64、128线程。RoundedGelu先执行原线性epilogue并舍入BF16，再用原Torch表达式做GELU，再舍入BF16。两处数值边界保留，FFN1中间结果不写global、GELU不再独立launch。FFN2继续调用原Torch。

两个独立进程，每点15轮baseline/新路径交替顺序；双方同样的固定cross缓存、prepared路径和Graph。表中为完整block CUDA Event中位数，不含profile；绝对延迟在进程间有漂移，全部原值公开。

|真实输入|基线引擎|进程1 基线→新路径 ms|进程1延迟下降|进程2 基线→新路径 ms|进程2延迟下降|
|---|---|---:|---:|---:|---:|
|short-B2-action-layer0-first|dense+cached+Graph|0.705664→0.695424|1.451%|0.705824→0.695072|1.523%|
|short-B14-action-layer0-first|guarded+cached+Graph|1.181184→1.144672|3.091%|1.183104→1.144864|3.232%|
|middle-B2-action-layer0-first|dense+cached+Graph|0.829792→0.820672|1.099%|0.835680→0.825984|1.160%|
|middle-B14-action-layer0-first|native+cached+Graph|1.428192→1.392256|2.516%|1.428096→1.393088|2.451%|
|long-B2-action-layer0-first|dense+cached+Graph|0.967680→0.958400|0.959%|0.912832→0.906112|0.736%|
|long-B14-action-layer0-first|guarded+cached+Graph|1.355552→1.320000|2.623%|1.257824→1.224320|2.664%|

六个中间量和完整block在两个进程均逐位一致，全部有效KV、全部metadata、输入、RNG和FFN1参数字节不变。B2下降0.736%–1.523%，B14下降2.451%–3.232%。这是小幅通用FFN组件收益，不是候选DAG调度或MegaTransformer协同结论。

## OBSERVED / UNKNOWN：计数与配置缺口

六点NCU记录L2 sectors、DRAM bytes、SM throughput和achieved warps-active指标。初六次用默认kernel replay/cache-control=all，其流量不能直接作为连续warm block流量；另两个short B2/B14 application replay/cache-control=none各实际运行6次，36次完整block host调用已计账。sectors不是requests；未测项目仍UNKNOWN。

本轮NCU命令漏显式关闭clock-control；本地help确认默认boost。是否实际改频UNKNOWN，原命令和结果保留。没有执行手动时钟/服务修改；以后bounded入口强制NCU显式--clock-control none与cache-control。全部NCU数据只作带此配置缺口的诊断，不进正式性能表。独立CUDA Event测量不在NCU内，但原空卡24%利用率异常仍在，因此不签正式独占性能。

资源元数据初次API返回active CTAs0发生在动态shared-memory opt-in之前，不能解释为物理occupancy0。单个真实B2输入执行两种已验证FFN1后重新读取：plain理论active CTAs/SM=2，plain/fused各154 registers/thread和73728 bytes shared memory。fused实测achieved occupancy未采，不由理论值冒充。

## 源码如何阅读

1. [baseline/block.py](../../../算子研究/wam_ops/baseline/block.py)第59行：保留原完整block数学顺序。第103行进入FFN，基线不改。
2. [optimized/prepared.py](../../../算子研究/wam_ops/optimized/prepared.py)：PreparedBlock固定slot/CFG；fill保留原shape投影；activate只临时切换合法shared-prefix和固定投影。退出还原。
3. [optimized/csrc/ffn.cu](../../../算子研究/wam_ops/optimized/csrc/ffn.cu)：RoundedGelu先BF16舍入再GELU；linear_tile在CTA内部调用CUTLASS mainloop/epilogue；没有host BLAS嵌入device、没有persistent队列。
4. [optimized/ffn.py](../../../算子研究/wam_ops/optimized/ffn.py)：FusedFFN1只接管ffn.net[0]并返回预分配BF16 activation，ffn.net[2]完全保留。install context退出删除实例覆盖，默认install_model未变。
5. [tools/profile_block.py](../../../算子研究/tools/profile_block.py)与[summarize_profile.py](../../../算子研究/tools/summarize_profile.py)是强基线和profile；[verify_ffn1.py](../../../算子研究/tools/verify_ffn1.py)是最终组件/完整block、状态和交错计时。

## RECOMMENDATION：本轮收口与下一步

强基线与可复用FFN1+GELU设备组件冻结；完整device FFN和persistent scheduler暂不具备原契约的推进条件。不要为了末位复现无限调参，也不将这约3%通用融合包装为论文主贡献。下一轮先围绕本次真实profile明确完整block DAG、可调度边界及FFN2原语的处理方案，再登记一个P_seq/P_sched实验；30层/完整chunk/质量逐级门仍在后续。

H1已有真实M与有限shape效率证据，但未证明唯一瓶颈。H2尚缺同一7-query逐个验证与集中验证；H3仅有通用FFN1组件增量，没有候选调度证据；H4未测有效推进或闭环。g、ΣT_round/Σg、chunk/episode speedup均UNKNOWN/N/A。

engineering_quality_screen=false；independent_test_complete=false；noninferiority_established=false；formal_performance_permission=false。旧FAILED、retention、TF/阈值、默认A_off与原tile/HW保护不改。

## 预算、终态与证据索引

GPU作业生命周期累计751.141秒（12.52分钟），含setup/profile/application重放/失败与清理，帽7200秒。CUDA-active另无独立计量。完整block host forward/capture/replay intents1093，原语调用617；全target/full30/新机器人episode均0。全部17个受监督作业终态，两个工具/兼容失败保留、无timeout；自有本机GPU进程全部退出。未用余额不自动续跑。

- [登记](REGISTRATION.json)、[输入身份](INPUT-MANIFEST.json)、[源保护](SOURCE-GUARD-RESULT.json)、[源清单](SOURCE-MANIFEST.json)。
- [强基线原始值/门检](profile-attempt2/data/RESULT.json)、[trace派生](PROFILE-DERIVED.json)、[旧候选源码复用](REUSED-CANDIDATE-SOURCE.json)。
- [完整device FFN失败](device-attempt2/data/RESULT.json)、[分块诊断](reduction-diagnostic/RESULT.json)、[serial诊断](reduction-serial-diagnostic/RESULT.json)。
- [新组件进程1](fused-ffn1-attempt1/data/RESULT.json)、[进程2](fused-ffn1-attempt2/data/RESULT.json)、[资源opt-in复核](metadata-post-optin/data/RESULT.json)。
- [NCU kernel计数](COUNTER-DERIVED.json)、[application对照](COUNTER-APPLICATION-DERIVED.json)、[NCU配置偏差](NCU-CONFIGURATION-DEVIATION.json)。
- [全部作业/清理/实际预算](RESOURCE-AND-STOP-REPORT.json)、[claim矩阵](CLAIM-EVIDENCE-MATRIX.md)。

第三方版本：[CUTLASS固定v3.5.1](https://github.com/NVIDIA/cutlass/tree/f7b19de32c5d1f3cedfc735c2849f12b537522ee)，项目内headers由fetch_cutlass.py取得；[Torch固定GELU源码](https://github.com/pytorch/pytorch/blob/0fabc3ba44823f257e70ce397d989c8de5e362c1/aten/src/ATen/native/cuda/ActivationGeluKernel.cu)。公开省略模型、原始PT、profiler大trace与二进制；主机保留并提供manifest，不能仅凭公开JSON宣称独立重算。
