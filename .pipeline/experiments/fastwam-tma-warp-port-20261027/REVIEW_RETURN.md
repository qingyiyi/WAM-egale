# 迁移完成：保持FastWAM eager字节精度

闭合时间：2026-10-27T05:49:32.227385+08:00。用户要求的旧TMA＋1T4M3E＋mbarrier已经迁入当前FastWAM完整单层persistent核。采用本轮版本作为后续研究源码；生产默认Torch，未接入投机队列。

## OBSERVED

- 10个线性投影均调用新局部流水：Self Q/K/V/O，Cross Q/K/V/O，FFN1/2。W0 TMA，W1–3 E，W4–7 MMA；输入双K64槽、结果槽用ready/empty/full/consumed mbarrier交接。FFN1的E保持BF16后GELU，写global ACT；FFN2和最终门控残差仍分阶段。
- 原阶段grid同步和局部初始化/排空CTA同步保留，没有声称完全无同步、DAG或DSM。E普通线程写回，不是输出TMA。
- 精度gold为固定官方Torch eager，不开compile；三层B1/B7的126隔离阶段＋6整层，以及完整30层×10step B1与30组B7的全部中间量/输出，uint8比较全一致。最大绝对误差0，相对L2误差0。KV/RNG只读和候选隔离通过。
- B7是七个真实canonical去噪状态的冻结诊断组合，不是当前训练draft树。原Torch B7对七次B1自身存在maxabs0.125/relL2=.001351607差异，本轮未修饰或据此放宽接受器。
- BF16递推未更改：输入GEMM的实际分段；B7输出16段；B1输出16 partial→FP32合并→BF16→bias→BF16。完整公式与交接见PIPELINE-CONTRACT.md。

## 同轮性能

|Batch|eager/自有（设备）|compile/自有（设备）|自有完整层µs|
|---|---:|---:|---:|
|B1|1.419×|0.659×|261.067|
|B7|0.959×|0.488×|591.755|

独占5090，固定真实layer15输入，10次去首留9；设备值为对称静态Graph控制测量，完整无外部Graph API见STAGE-TABLE.md。官方compile部署路径继续作为强对照，不将其数值融合差异误判为本轮精度失败。

历史前轮自有设备耗时B1=280.811µs、B7=667.100µs；本轮=261.067/591.755µs，描述性下降约7.0%/11.3%。这不是同轮旧新交错试验，不据此宣称统计显著。仍未胜过官方compile。

## HYPOTHESIS / UNKNOWN / RECOMMENDATION

- 剩余优先定位B7 Cross输出、FFN2和Cross KV。分段提交次数、有效CTA数量与库split-K并行度是可核验候选原因；本轮未做硬件counter因果归因，不称已证明带宽/占用率瓶颈。
- 当前TMA是同CTA输出tile内流水，B7输出仅32输出tile、每tile16段；没有移植旧模型错误的7×K2048，也未擅自把serial BF16改为一次FP32归约。
- 本轮只完成迁移、精度、有限单层计时；全机器人、真实draft接受翻转、训练/阈值、完整policy speed和Robo形状不在本结果范围，四科学权限false不变。
- 建议下一轮先针对B7输出16段设计有序提交的可并行调度，维持本轮已核对的精确递推；不要先扩大整层DAG。该建议尚未实现。

## 证据与入口

- `validate-attempt01/{STAGES,WHOLE,STATUS}.json`：隔离/三层精度。
- `policy-validation01/{B1-ALL,B7-ALL,ISOLATION,STATUS}.json`：330组raw-byte与状态验证。
- `benchmark-attempt01/{STAGES,WHOLE,STATUS}.json`：全部原始样本/边界/精度。
- `STAGE-TABLE.md`：21完整阶段、tile、warp、速度；分阶段不能加总代替整层。
- `source-final/optimized/csrc/{fastwam_persistent,fastwam_tma_pipeline}.cuh`：固定源码；活跃代码仍原optimized目录，直接迭代。
- `binary/fastwam_sm120_persistent_cuda.so`：本轮独立二进制；原baseline二进制在前轮归档。
- `RESOURCE-AND-STOP-REPORT.md`：实际预算/清理，原始JOB.json不改。
