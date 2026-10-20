# 简易版选定与单层阶段对齐结果

**采用简易版作为唯一活动整层优化入口。** 真实输入有限精度门通过；两独立进程交错计时未观察到稳定退化。复杂版活动入口/开关已移除，历史源码、原FAILED和原始计时保留。baseline、投机代码、原tile与保护HW未改。

## 普通完整单层计时

远端A100-SXM4 GPU0启动前无其他GPU0计算进程；GPU1独立跑RoboTwin。固定B14 Q16，真实短/中/长KV。每进程每格每引擎6轮预热、18普通交错样本；profiler另跑，不进入表。表给两进程中位数，不合并时钟/热状态差异。uncached/eager，输入与cache准备排除；不是强cached/Graph、30层、chunk或E2E。

|KV|进程|native ms|复杂版 ms|简易版 ms|简易/复杂|简易/native|
|---|---:|---:|---:|---:|---:|---:|
|short|1|2.1253|4.1226|4.0975|0.9939|1.9280|
|middle|1|2.4046|4.4969|4.4787|0.9959|1.8625|
|long|1|2.8411|4.8987|4.8655|0.9932|1.7126|
|short|2|2.6209|4.8737|4.8420|0.9935|1.8474|
|middle|2|2.3995|4.4987|4.4696|0.9935|1.8628|
|long|2|2.8376|4.8871|4.8644|0.9954|1.7142|

简易版较复杂版中位数低0.4%–0.7%；只据本有限面板选择更简单实现，不宣称严格性能等价定理。短KV的跨进程native/自有时间有明显变化；原首轮第二进程短KV曾为+5.28%，原全部样本及失败保留。此次新增预热/交错后两进程均无稳定退化，未删除旧不利记录。

仍未对齐原版：完整层简易/native约1.85–1.93（短）、1.86（中）、1.71（长），仍有约71%–93%额外耗时。

## 精度、输入与状态

9种真实签名：B2 action Q16、B14 action Q16、B2 video Q128各三KV。72完整block验证；最终输出/23可配对中间量、18个合法KV转移、重复结果、输入和CPU/CUDA RNG均有限逐位通过，最大绝对误差0。两进程普通计时同输入新旧版输出/中间量/有效KV也通过，六profile各一个compute kernel。有限fixture不等于所有输入证明、完整模型/机器人质量或正式性能准入。

外部审计修正：原脚本把logical_slots的无效尾部也逐字节比较；LONG-DIAGNOSTIC确认有效2144槽相同，剩余16槽未使用。新脚本比较有效范围；旧FAILED原样保留。核心tile visits审计删除，字段明确NOT_COMPILED，不伪造exact-once证书。

## 逐阶段结果与解释

完整90行矩阵见[STAGE-RESULT](STAGE-RESULT.md)和[CSV](STAGE-MATRIX.csv)。A原版阶段/B简易独立阶段/C简易嵌入整层，原始每样本保留在simple-stage-process1/2-RESULT.json。29有效阶段×3KV×2进程均有限逐位通过，cache_insert在B14只读路径N/A。自有阶段统一108CTA/256线程，未逐阶段优化grid。

重点：cross K/V GEMM独立只慢约9%–10%；FFN1慢约83%–84%；cross-K RMS慢约3.4倍、GELU约6倍、table约5倍、layout约21–36倍。整层嵌入另有资源/缓存/调度差异，不能把所有差距归因于global barrier。插桩改变local资源，C/A只作描述，不代替普通计时。

优先pointwise并行与B14只读layout，再cross-K RMS、FFN1/cross-attention；最后ready跨阶段填尾波。这些是有profile依据的下一步，不是已完成收益。当前未实现性能完全对齐。

## 简易版读法

1. [whole_layer.py](../../../../算子研究/wam_ops/optimized/whole_layer.py)：绑定原参数和当前KV。
2. [whole_layer.cu](../../../../算子研究/wam_ops/optimized/csrc/whole_layer.cu)：五段注释展示单层DAG；begin_stage/claim_tile/finish_tile展示队列。
3. GEMM设备主循环在cutlass_gemm.cuh与gemm_pipeline.cuh；FA2在fa2_stage.cuh。归约/舍入/有效mask和同步是算法内容，保留。

删除重复输入检查、tile visits/stats、未启用的FFN-ready分支及旧独立pipeline入口；外部脚本承担结果核验。主文件展开格式并加中文注释，所以行数不作为“更简单”的指标。原相同数学数据流、单kernel和108常驻CTA保留，阶段之间仍grid.sync。

[资源与失败](RESOURCE-AND-STOP.md)、[来源与远端/本地核对](FINAL-AUDIT.json)、[选定记录](SIMPLE-SELECTION.json)。四项quality/test/noninferiority/formal-performance仍false。
