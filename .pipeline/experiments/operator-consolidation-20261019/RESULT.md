# 两份算子代码整理、实际30层精度与计时结果

更新：2026-10-19T11:14:53.440262+08:00

代码整理和有限整模型精度已完成。性能数据已采集，但三个attempt均在资源检查发现外来GPU进程，仍为FAILED/PARTIAL；不签稳定独占性能证书。登记的150次full30调用已用完，自有GPU进程均已退出，不追加试跑。

## 代码与清理

只维护`算子研究/wam_ops/baseline`和`optimized`两份主实现，共用`install.py`及`tools`。baseline完整block按原Torch顺序；optimized复用同一block、替换独立CUDA SDPA；尚无新MegaTransformer调度。block forward与固定ca5f37e逐AST一致，删除未使用的其他cache模式。CUDA数学源与原对齐v2逐字一致，按同精度编译选项重新构建，仅改模块/文件命名。

已删除16个无投机调用的旧算子目录、26444694字节，以及新重复build5044712字节。308源在Git a0ed2d8e49403cdca8d87afd40b21dd19595710c中保留；所有旧实验结果/失败保留，不留另一套活动实现压缩包。

1875份投机源/配置/运行依赖前后逐字未改。原tile仍被投机shared-prefix的prepare_source.py引用，原位只读保留。投机服务、训练、接受器、调度、远端评估未修改。

## 精度与状态

三attempt共130项检查全部通过、max_abs=0；其中两个独立进程完成全部六个真实block点与六个实际30层点，另有首轮短KV部分。130含参考与计时后检查，不是130独立输入。实际调用共150次完整30层、168次独立block。

- 六个真实action block0：B2/B14×attention总K288/1152/2160，对原版和历史GPU gold逐位一致。
- 实际30层：B14三长度历史GPU gold逐位一致；B2三长度使用真实verifier同root CFG0/7行、普通B2临时cache路径，明确为派生诊断而非保存的原始顺序B2轨迹。
- 合法write1/2：short B2分别检查输出及最终有效KV/metadata，不能只要求cache不变。
- 所有有效KV、全部metadata、输入、当前CPU/CUDA RNG通过；模型安装/还原、参数对象/地址通过。无FSDP分片，不将本项扩展成未测的FSDP路径证书。

原生产CUDA stride/alias/RNG未持久化；视频完整block、所有sigma/边界、完整policy/接受/scheduler路径未由本项新增证明。原M22/M24有限证据和旧失败保留，不自动继承质量结论。

## 全部计时

下表是两个完整矩阵中的模型中位耗时比，全部标为探索性。run A为三臂三轮矩阵，run B为原版/baseline三轮对照；run B仍做三臂数值。两个进程加载同一checkpoint并在计时前核对同一输入的精度。B14为保存真实输入，B2派生来源如上。

| 输入 | A baseline/原版 | A optimized/原版 | B baseline/原版 |
|---|---:|---:|---:|
| short-B2 | 0.9141 | 0.9254 | 1.2380 |
| short-B14 | 0.9567 | 1.0093 | 0.8284 |
| middle-B2 | 0.9374 | 0.9136 | 1.0315 |
| middle-B14 | 0.9922 | 0.9901 | 0.9244 |
| long-B2 | 0.9809 | 0.9097 | 1.0348 |
| long-B14 | 1.0341 | 1.0471 | 1.0045 |

可见B进程short-B2为1.2380、short-B14为0.8284，连同block middle-B2的0.6197，表明测量波动明显。不能挑更快点、合并后宣称加速或稳定对齐，也不能仅据受干扰计时判代码本身更慢。当前可用结论：有限输入精度/状态复现完成；性能稳定验收仍PENDING。若后续提供新有界作业与连续空闲窗口，只补计时；当前预算不重置。

GPU worker生命周期合计137.306秒，含加载、失败与清理；全作业另含CPU准备和等待，未将其说成CUDA active时间。三轮均保留自然非零退出收据，未补造exit0。正式机器人E2E与sum(T_round)/sum(g)不是本项指标。

## 原始证据

- [数值全表](NUMERICAL-RESULT.csv)、[全部计时原始样本](TIMING-ALL-SAMPLES.csv)、[每attempt/每点汇总](TIMING-SUMMARY.csv)
- [首轮短KV](process-1/RESULT.json)、[完整矩阵A](continuation-1/process-1/RESULT.json)、[完整矩阵B](continuation-2/process-1/RESULT.json)
- [最终预算与清理](FINAL-RESOURCE-AND-STOP-REPORT.json)、[删除清单](CLEANUP-RESULT.json)、[路径迁移](PATH-MAPPING.md)
- [源码身份](SOURCE-IDENTITY.json)、[投机源保护](SPECULATIVE-SOURCE-GUARD.json)、[CPU核对](CPU-RESULT.json)
- [简化源码入口与讲解](../../../算子研究/README.md)

engineering_quality_screen=false、independent_test_complete=false、noninferiority_established=false、formal_performance_permission=false。
