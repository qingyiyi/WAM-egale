# 连续TMA输入流水：精度通过，净性能未改善

终态 **CLOSED_REJECTED_PERFORMANCE_RESTORED_A**。评估TMA-CONTINUOUS，当前实际保留 **SPLITK-M64-FFN2**，此前FFN2收益保留。

## OBSERVED：实际实现

只改变三个小投影（Self输出、Cross Q、Cross输出）的输入执行：每投影96个输出tile，M64×N128、原8warp MMA，两个K64输入槽；48个K64块的输入ring只初始化一次，跨9×K320＋K192十个数值段持续预取。两组寄存器fragment在当前MMA前预取下一组。仍保留每段MMA零起点、原部分和/上段BF16/bias顺序及8warp后处理，不引入后处理warp重叠。

原CUTLASS warp MMA数学和布局保留，global→shared改TMA，段内及段间访存/同步日程变化；不是仅替换一条搬运指令的纯单因素比较。输入和结果为独立shared区域，原FFN2、SelfQKV联合队列、attention、norm/RoPE函数未改。

三份真实B14/Q16/block0、KV288/1152/2160的26中间量＋输出、normal/debug、有效KV/输入/RNG有限逐位通过。每tile实际发起48次A/48次B加载、48次等待、十次合并，三个投影96tile全部符合计数。CPU顺序模型和源码自审见CPU-AUDIT.json及SOURCE-REVIEW.md；本轮没有独立代理审查，不冒充独立复核。

|完整层，两个独立进程同场对照|短KV|中KV|长KV|
|---|---:|---:|---:|
|候选相对A耗时增加|1.384%|1.125%|1.065%|
|原Torch/候选|1.1355×|1.6060×|2.0202×|
|原Torch/当前保留A|1.1512×|1.6240×|2.0417×|

两个进程共六个KV观测均退化约1.04%–1.52%，因此本轮不采用。四臂eager无Graph、同GPU/输入/预计算CrossKV、交错顺序；冷构建另列。完整正常forward包含六个TMA tensor map的构建成本。当前A对guarded短快1.485%、中慢0.300%、长慢2.979%，不夸大中档小差异。

分时插桩诊断中，TMA三个投影约0.0619–0.0645ms，原路径约0.0594–0.0615ms，慢约3.3%–4.9%。这仅定位改动投影的区间，不能代替同场无插桩完整层判定。候选插桩扰动绝对≤0.279%，当前≤0.340%。

Shared可用性不是本次失败原因：候选100352B、A100096B，仅增256B；输入裸双槽48KiB＋独立34KiB结果能放进原整层最大workspace。两者均170CTA、1CTA/SM、256线程、255寄存器/线程、784B静态local/线程。静态属性不能替代动态访存计数。

## HYPOTHESIS / UNKNOWN

TMA输入与当前计算可并行的机制已实现，实际重叠比例尚未测量。每K64仍有输入ready和槽复用两次CTA同步，每个数值段仍有结果转存/合并同步；后处理由8warp顺序执行，未与下一段MMA形成专用warp重叠。相对原CUTLASS已有三级cp.async和寄存器流水，新增TMA机制未显现净收益。

尚未定量区分TMA等待、shared→register、MMA、结果合并、局部barrier和host map构建的贡献。不能以此次结果断言TMA硬件无效、输入总不是瓶颈，或直接宣称后处理是唯一根因。输入ring支持跨段预取不代表全部加载延迟已被隐藏。

## RECOMMENDATION

当前保留原投影及已有效FFN2 split-K。若继续，先在现有候选中软件分量计时，检查输入等待与同步是否抵消收益；只有输出合并开销足够明显时再实现后处理warp交接。仍须与原高效主循环比较，不直接进入大DAG/统一warp分工。后处理重叠尚未实现或作为新作业启动。

## 回退、交付与资源

A的五份源码及实际public入口已恢复，默认/非默认流再次有限逐位/状态通过；候选源码与正常/profile二进制保留。回退后当前17阶段也已本轮重测，完整表在STAGE-TABLE.md；候选完整表在reports/TMA/STAGE-TABLE.md，均已在对话直接打印。RETAINED诊断的旧GRAIN参考列不混入本轮A主表。

两个构建和六GPU进程均EXITED_OK，无编译/运行失败或timeout，性能候选拒绝1个。完整block11469/15000次，GPU worker进程生命周期51.083秒不是CUDA active。保守本轮计15分钟，累计33286.413/36000秒，余额2713.587秒（约45.2分钟），旧预算起点/截止不重置。新增约37MiB，低于1GiB。

16个owned PID与出生身份均已退出，5090显存0；六份源码/二进制回本机byte一致。实例未关机。双A10010:02只读见原RoboTwin模型/模拟上下文和巡检正常，本轮不改队列、不重算SR。

四科学权限及production_admitted均false，有限单层证据不外推30层、在线接受或机器人E2E。

## 固定证据入口

- [当前/候选身份](SELECTED.json)、[当前完整表](STAGE-TABLE.md)、[候选全表](reports/TMA/STAGE-TABLE.md)
- [候选源码](candidate-source/csrc/whole_layer.cu)、[两进程结果](reports/TMA/SUMMARY.json)
- [资源](RESOURCE-AND-STOP-REPORT.json)、[清理](OWNED-CLEANUP.json)、[byte回传](RETURNED-BYTE-CHECK.json)
