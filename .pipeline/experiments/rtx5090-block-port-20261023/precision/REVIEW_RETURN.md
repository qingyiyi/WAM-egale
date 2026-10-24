# RTX5090单层精度修复回传

终态：**FINITE_EXACT_PASS_RESEARCH_ONLY**。更新：2026-10-23T20:33:39.096761+08:00。

在当前RTX5090 / Torch2.9.0+cu130上，三份真实B14 block0输入的26项中间量和最终输出，均与未改数学实现、未改后端精度设置的原Torch逐位一致。默认加载入口的两个独立进程，分别使用默认流与自建CUDA流，共156次完整block调用；未使用CUDA Graph、未计时。三份输入反复运行不等于156个独立输入。

|KV|输出不同元素|最大绝对差|相对L2|26中间量|有效KV/输入/RNG|
|---:|---:|---:|---:|---|---|
|288|0|0|0|逐位一致|通过|
|1152|0|0|0|逐位一致|通过|
|2160|0|0|0|逐位一致|通过|

## 查明并修正的差异

1. 原迁移版沿用A100的两段累加。当前5090原Torch的六个224×3072→3072投影实际采用split-K10：九段320、尾段192；FFN2采用七段2048。分段之间写回BF16，改变分段会改变数值。核验来自真实原Torch kernel名称、grid及实际修正结果，见`results/partitions-data/RESULT.json`和对应trace。
2. 最后分段的FP32加法次序也影响舍入。SM120改为先合并上一段BF16结果，再加bias，随后BF16舍入。投影修改消除了残留的2–4元素分歧；FFN2相同修正消除了长KV最后1个元素的差异。SM80分支原顺序不改。CUTLASS本地bias epilogue源码提供对照，最终判断以真实输入逐位结果为准。
3. 公共Python入口的Cross KV缓存改为先按原B14形状执行投影/RMS，再去重输出，避免先压缩输入改变GEMM舍入。确认脚本独立调用此接口生成缓存，没有把参考中间张量塞回自有输入。
4. FFN2为保持七段有序提交，等待完整ACT行块就绪后，由原四MMA warp顺序处理七段；输出tile仍128×128，共48个，每个七段。没有改变原Torch精度选项、权重、阈值，也没有修改Attention数学实现。

分段算式：中间段 r_i=BF16(P_i+FP32(r_(i-1)))；最后段 r_last=BF16((P_last+FP32(r_prev))+bias)。每个P由当前MMA段累加得到，不能把这条式子解释成任意GPU/库版本都采用同样分段。

## 失败和修复路径

- projection10：尾段仍按320迭代导致非法访存，原FAILED保留；按有效192修复迭代次数。
- projection10-tail：投影大幅接近但FFN2仍两段，完整层未过。
- ffn7：短/中最终输出一致，长仍有352元素差异；中间投影仍未过。
- bias-last：三档输出一致，长FFN2仍有1元素差异；未提前判通过。
- ffn-bias-last：三档27个比较字段全部通过；随后默认加载入口两进程重复核验通过。

各阶段程序EXITED_OK只表示进程正常结束，算法数值终态以`RESULT.json`为准。没有改写旧FAILED。

## 保留版本、范围与下一步

当前仅5090独立研究副本保留`ffn-bias-last`；原迁移和A100二进制保留。170CTA×8warp，shared100096B，255寄存器/线程，静态local784B；这不是动态HBM测量。

当前有限证据覆盖三份B14真实输入、readonly历史、缓存Cross KV、默认与非默认流；未覆盖B2、任意输入/层、未预计算分支、在线cache写入、30层、完整机器人闭环。四项科学权限及production_admitted仍false。

**修复后性能未测。此前1.22/1.72/2.14倍来自数值未通过版本，不能挪给本版本。** 下一步是在当前已对齐源码上重新编译阶段插桩版，先核对插桩/正常版输出，再做原Torch双方eager的短中长KV完整层和阶段对比；旧profile二进制不对应本版本，不直接使用。无需再做结构搜索。

## 固定入口

- 当前本机源码：`/rjs/cxz/论文探索/.pipeline/experiments/rtx5090-block-port-20261023/workspace/算子研究/wam_ops/optimized`。
- 当前本机独立二进制：`/rjs/cxz/论文探索/.pipeline/experiments/rtx5090-block-port-20261023/precision/build/ffn-bias-last/wam_whole_fa2_cuda.cpython-312-x86_64-linux-gnu.so`。
- [源码改动](CHANGE.patch)、[身份与回传核对](SELECTED.json)、[完整阶段表](STAGE-TABLE.md)。
- [默认流](results/confirm-default-data/RESULT.json)、[非默认流](results/confirm-custom-data/RESULT.json)、[FFN队列终态](FFN-STATE-AUDIT.json)。
- [资源](RESOURCE-AND-STOP-REPORT.md)、[进程清理](OWNED-CLEANUP.json)。源码9文件与远端返回副本逐字节一致；未计算SHA。
