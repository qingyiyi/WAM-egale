# FFN局部TMA流水：有界结果

终态：CLOSED_NEGATIVE_RETAIN_GRAIN；2026-10-23T23:45:25.867780+08:00。已实际完成三个候选，实现/编译/三KV精度/两进程完整层与阶段测量，不停在规划。当前保留GRAIN-M64-FFN2，三个新候选均不采用；原INPUT/EPILOGUE/KREADY源码与二进制保存，未修改旧科学门或baseline/投机代码。

|候选|实现变化|精度/状态|相对GRAIN完整层延迟：短/中/长|决定|
|---|---|---|---|---|
|INPUT|独立W1 TMA producer、四MMA consumer；输入full/empty，仍8warp串行提交|三fixture有限逐位及状态通过|+3.23%/+2.88%/+2.70%|不采用|
|EPILOGUE|在INPUT上增加独立单FP32输出槽，W2/W7按序提交，与后段计算并行推进|三fixture有限逐位及状态通过|+4.03%/+3.79%/+3.52%|不采用|
|KREADY|在EPILOGUE上按2×7个K2048片段ready，固定生产优先与七段提交|三fixture有限逐位及状态通过|+3.80%/+3.54%/+3.61%|不采用|


## OBSERVED

- 每候选均覆盖三份真实B14/Q16 block0输入：KV288/1152/2160。26个中间量＋最终输出、普通/debug输出、有效KV、输入、CPU/CUDA RNG有限逐位通过；计时前后再次核验。只读block重放不是完整机器人质量或30层验证。
- KREADY新增14个ready计数均为13个8和最后1个16；原四组生产/提交审计保持。新ready只属于FFN局部，不恢复整层DAG。
- 两独立进程的每一档新候选都慢于同场GRAIN；完整层计时来自正常二进制，profile分开。本轮未以target调用数或接受率代替延迟。
- 当前保留GRAIN在最后同场原Torch/自有为1.0892/1.5417/1.9568倍，仍慢guarded约4.03%/5.56%/7.45%。不能把现有缓存/普通融合收益认作本轮TMA收益。
- M64/N128/K64下两个输入槽48KiB＋单FP32输出槽34KiB≈82KiB（不含metadata/对齐），整层实测100352B=98KiB；原GRAIN100096B=97.75KiB。新候选均170CTA、1CTA/SM、255寄存器/线程、静态local784B/线程。容量放得下，故没有触发缩tile备选，没有尺寸搜索。
- 新正常kernel的编译统计为232B stack frame、756B spill stores、872B spill loads；GRAIN为232/764/900。它们是静态编译统计，不能当作动态local/HBM流量，更不能据此断言性能根因。
- 诊断生成器首次未适配ffn_ready声明，KREADY profile编译失败（0 GPU block调用）；已保留失败源和日志，一次修正后完成。正常算术源码未因该修复变化。
- GRAIN源码已恢复，两种实际public流入口再次有限通过；6份源码/二进制回传逐字节一致。所有本轮owned进程自然退出，5090显存0MiB；未关闭实例，双A100原RoboTwin队列未改。

## HYPOTHESIS / UNKNOWN

输入TMA/full-empty握手、缺少原CUTLASS相同寄存器预取组织、两warp后处理吞吐及单输出槽回压都可能影响差距。本轮不能分离其各自因果贡献。实现允许producer、MMA与epilogue独立推进，但没有硬件计数器或逐角色活动证据证明实际Tensor Core/CUDA Core同周期重叠比例；不把代码结构当作该比例的测量。

KREADY相较前一轮EPILOGUE的细微差别来自不同进程组，未作为严格同场单因素机制收益。已确定的是：三种新候选都未胜过各自同场GRAIN。未证明TMA普遍无效，也未证明缩tile一定更快。

## RECOMMENDATION

保留GRAIN，结束本轮，不再追加组合。下一步先隔离同M64的原CUTLASS主循环与新producer/consumer的等待、寄存器预取和提交成本；若研究更小M32，单因素比较其168个输出tile相对于当前96个的并行度与重复权重读取代价。M32尚未实现/测试，此建议的理由是任务并行度，不是本轮shared超限。不要直接开启尺寸网格或完整DAG。

## 固定入口

- 当前采用源码：`/rjs/cxz/论文探索/.pipeline/experiments/rtx5090-block-port-20261023/workspace/算子研究/wam_ops/optimized`；默认独立模块`wam_whole_grain_cuda`，`production_admitted=false`。
- 原A（本轮GRAIN）冻结源码`A-source/`、独立二进制`A.so`，真实PyInit名`wam_whole_grain_cuda`。
- INPUT：`INPUT-source/csrc/whole_layer.cu:492`，二进制`build/input/wam_whole_pipe_input_cuda*.so`。
- EPILOGUE：`EPILOGUE-source/csrc/whole_layer.cu:565`，二进制`build/epilogue/wam_whole_pipe_epi_cuda*.so`。
- KREADY：`KREADY-source/csrc/whole_layer.cu:565`；片段发布`:652`，二进制`build/ready/wam_whole_pipe_ready_cuda*.so`。
- 三套原始结果分别`results/<候选>-precision-data/RESULT.json`、`results/<候选>-measure1-data/RESULT.json`与`measure2-data/RESULT.json`；完整候选表`reports/<候选>/STAGE-TABLE.md`，各自采用决定`DECISION.json`。
- 当前保留表：[STAGE-TABLE.md](STAGE-TABLE.md)，固定选择：[SELECTED.json](SELECTED.json)，回传：[SOURCE-RETURN-MANIFEST.json](SOURCE-RETURN-MANIFEST.json)，资源：[RESOURCE-AND-STOP-REPORT.json](RESOURCE-AND-STOP-REPORT.json)，清理：[OWNED-CLEANUP.json](OWNED-CLEANUP.json)。不执行SHA任务。

四项科学权限与production_admitted保持false。三份有限fixture不升级为任意输入、完整chunk、机器人闭环或正式新TF性能证书。
