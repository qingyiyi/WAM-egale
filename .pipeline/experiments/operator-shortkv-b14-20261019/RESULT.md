# 短KV/B14核查：复现性能与融合收益分列

本轮已完成源码/表格核查和有限真实输入重放。三个有界作业均终态：两次被他人新GPU任务打断，最后一次自然空闲等待到期、未建CUDA context。新增计时为PARTIAL，未修改数学内核，不能声称所有旧慢点已解决。

## 先核对2.5%的含义

用户关心“自有实现比原Torch慢”，以及B14全部差异约2.5%是否合理。上次M26的B14 **2.45%–3.23%是耗时下降，即更快**，来自可选FFN1+GELU融合；该版本保留Torch FFN2。B2下降0.74%–1.52%。它不是未融合复现版的退化，也不是数值误差。

统一定义延迟比`T_new/T_ref`：小于1为快。下面两类对照必须分开，不能用融合收益抵消复现慢点。

### 默认未融合复现版 / 原Torch

复用M25同输入、未缓存eager、两个独立进程。表为**耗时变化百分比，负数更快**，区间覆盖两个进程。

|输入|完整block0耗时变化|完整30层耗时变化|
|---|---:|---:|
|短KV B2|−2.50%～−2.24%|−4.65%～−1.08%|
|中KV B2|−0.84%～−0.51%|−2.37%～+2.18%|
|长KV B2|−0.15%～+1.62%|−0.68%～+0.65%|
|短KV B14|−1.82%～−0.03%|−2.65%～−0.16%|
|中KV B14|−0.05%～+0.13%|−3.32%～−0.70%|
|长KV B14|+0.20%～+1.19%|−0.82%～+0.81%|

逐点原值见[REPRODUCTION-VS-TORCH.csv](REPRODUCTION-VS-TORCH.csv)。本表没有显示三个B14均慢2.5%以上；不能据此取消旧54点中的异常。B2 full30仍是实际verifier CFG0/7派生诊断，非原始顺序轨迹。M25的严格±2.5%筛查和NVML异常保留，不追溯改PASS。

### 可选FFN1+GELU / 同配置未融合强基线

双方使用相同条件缓存、attention引擎与Graph，均为完整block0静态重放。表为**延迟下降，正数更快**。

|KV|B2延迟下降|B14延迟下降|
|---|---:|---:|
|288|1.45%～1.52%|3.09%～3.23%|
|1152|1.10%～1.16%|2.45%～2.52%|
|2160|0.74%～0.96%|2.62%～2.66%|

见[FUSION-VS-MATCHED-UNFUSED.csv](FUSION-VS-MATCHED-UNFUSED.csv)。六点×两个进程的中间量/完整block输出/有效KV/metadata/输入/RNG/参数字节均过，max_abs=0；不是允许2.5%的数值误差。融合省去FFN1结果一次写回、GELU一次读取及单独launch，B14该中间量是B2的7倍；这是逻辑工作量解释，不能直接当实测DRAM节省。完整30层/完整policy的新融合未验证，默认optimized也未自动启用它。

## 新增A/A与直接源码核查

Torch_A与Torch_B调用完全相同操作；各臂18轮轮转，Graph核心每次8个相同操作，profile单独运行。新增同口径缓存Graph完整block中，短B14自有/Torch=0.99897、中B14=1.00139；未复现统一慢2.5%以上。短B14未缓存eager为1.02179，而同一Torch的A/A为1.03045；中B14未缓存为1.00490。只能说明本次eager波动不可忽略，不能证明所有旧慢点都是噪声。

短B14核心Graph自有/Torch：self约1.0061、cross约1.0116；约1%的微小差异仍保留。旧M22短B14核心约慢0.4%–1.2%，完整block两个旧点慢6%–8%，旧最大34.37%其实是long/B2/layer8；全部原值保留。只有9份代表输入保存，不能声称本轮独立重放了旧54个中间量。

固定Torch `0fabc3ba44823f257e70ce397d989c8de5e362c1`与独立CUDA的标准tile同为Q128/K64，split同为Q64/K128，选择公式一致。B14每实例仍Q16；两种短B14实际kernel均grid[1,14,24]、128线程、255 registers/thread、65536字节shared memory。profiler估算occupancy=0字段不作为有效物理结论。baseline block forward与原forward AST相同，wrapper没有隐藏QKV拷贝。

本地nvcc12.4与Torch wheel CUDA12.6不同，SASS及其约1%差异贡献UNKNOWN。未定位到新增算法工作量的证据；不在此时盲改tile、增加fast-math或降低精度门。

## 是否合理与下一步

性能差异需要先分方向与比较对象。融合更快约3%是优化信号；同路径复现若三种B14均稳定慢超过2.5%，应按系统性回归处理，不能仅称“正常”。旧2.5%是登记工程筛查，不是通用统计定理，原筛查未全过仍保留。后续新测量要有可用窗口、A/A及跨进程重复；当前没有完整新矩阵，不额外宣称稳定性能放行。

保留已通过的未融合精度基线与可选融合组件；需要处理旧layer29/last异常时，先补足同源完整输入身份，再做有限复测，不以代表点代替全范围。此次不启动新融合/大调度/训练/机器人episode。

## 证据与资源

- [新逐点计时](NEW-DIAGNOSTIC-TIMING.csv)、[前两次派生值](FIRST-TWO-ATTEMPTS-DERIVED.json)、[旧慢点原轮次](OLD-SLOW-ROUNDS.json)。未完成行仍留原始RESULT。
- [源码核查](TORCH-SOURCE-AUDIT.json)、[开始源码](SOURCE-MANIFEST-START.json)、[最终源码及工具版本边界](SOURCE-MANIFEST-FINAL.json)、[1875投机文件/基线/保护文件复核](SOURCE-GUARD-RESULT.json)。
- [登记](REGISTRATION.json)、[所有作业/终态/清理](RESOURCE-AND-STOP-REPORT.json)。新增GPU作业生命周期约665.685秒，包含等待/准备/失败/清理；GPU阶段墙钟约31.145秒，CUDA active UNKNOWN。6180 attention、918 block host/capture/replay intents；full30/新episode均0。三个自有进程全部退出，余量不自动复用。
- [远端CPU只读巡检快照](REMOTE-STATUS-READONLY.json)，原Object后继自动接续与截止不改。

engineering_quality_screen=false、independent_test_complete=false、noninferiority_established=false、formal_performance_permission=false。旧FAILED与原tile不改。
