# FFN两阶段同步版：采用S，独立A保留

## OBSERVED

本轮仅在FFN1+bias/BF16/GELU写global ACT后增加grid同步，再执行原FFN2七段K2048与最终gate/残差。保留主体/尾tile、warp职责、七段FP32部分和及逐段BF16递推；没有新增DSM或统一warp比例。源码函数改名action_ffn_stages以反映实际阶段。

两独立进程×三真实KV fixture全部通过：27处输出/中间tensor逐位、七段partial/running逐位、有效KV/输入/RNG、非默认stream及profile输出。更新后的公开Python入口又逐一通过正常/debug输出及状态检查。有限fixture通过不等于任意输入证明，也不开放机器人质量/正式性能门。

完整层相对原A三个KV均改善，两个进程各自改善约1.74%–2.30%；满足事前≤3%回退筛选，并采用S作为下一轮易分析基线。170CTA、100352B shared/CTA、255regs/thread、784B静态local未变；profile静态local为792B，其整层扰动另列SUMMARY。

分开观察后FFN1+GELU只有原Torch速度的0.797–0.805倍（约慢24%–25%）；FFN2+gate/残差为1.027–1.033倍，已接近/略快于同边界原Torch。Self输出、Cross Q、Cross输出投影仍约0.84–0.88倍。完整18阶段表与原始时长见STAGE-TABLE.md、SUMMARY.json。

## 原因与UNKNOWN

这次受控修改说明当前按行组交接的FFN跨阶段重叠并未带来净优势；新增grid同步的S没有性能退化，不能简单把同步数量当瓶颈。源码同时改了函数名，编译器可改变布局；尚未把改善严格分离成某一种等待或缓存效应。

FFN1只有120个生产任务，少于170个常驻CTA；主块包含部分有效行，GEMM后FP32结果分两半经过shared、CTA屏障、bias/BF16/GELU再写ACT。这些是可核对的并行度/交接约束，但当前阶段区间尚未分离纯MMA、输入搬运、GELU、输出转换和屏障尾部，不能据此声称其中任一项就是全部慢因。无新增硬件计数器证据，不声称HBM或Tensor Core利用率变化。

## RECOMMENDATION

下一步只聚焦FFN1：先对照GEMM主体与GELU/写回成本，并区分主体和尾块、CTA实际任务分配；再根据结果决定是否改变分块或输出交接。保留FFN2现有七段实现及阶段全局同步，不同时引入DSM或完整DAG。三个小投影作为后续第二优先级。

## 固定入口

- 当前源码：../../rtx5090-block-port-20261023/workspace/算子研究/wam_ops/optimized/csrc/whole_layer.cu（实际绝对路径见SOURCE-AND-RESULT-ENTRY.json）。
- 本轮冻结源码whole_layer.cu；旧A为A-whole_layer.cu及上级A.so，公开入口旧版A-whole_layer.py保留。
- 新二进制build/normal，独立观察器build/profile；make_profile.py可再生成18阶段观察器。
- 原始结果measure-1/RESULT.json、measure-2/RESULT.json；七段数值measure-1/numeric-candidate-*.pt；PUBLIC-ENTRY-RESULT.json核实际Python入口。
- JOB.json保留本轮起点/上限；原失败与旧窗口不修改。资源/清理另见RESOURCE-AND-STOP-REPORT.json。
