# FFN四组依赖流水：有限精度通过，整层小幅正向信号

2026-10-22完成。用户批准在当前optimized试验FFN1+GELU到FFN2+残差的局部流水；baseline、投机侧、原tile及四科学权限不变。最终R2采用为研究入口，本机仅复制远端验证二进制、未另做GPU计时。

## OBSERVED：同轮原Torch对照

|KV|此前阶段版相对原Torch|R1相对原Torch|最终R2相对原Torch|R2较此前阶段版延迟变化|
|---|---:|---:|---:|---:|
|288|1.1607×|1.1513×|1.1646×|-0.342%|
|1152|1.6748×|1.6644×|1.6808×|-0.357%|
|2160|2.7295×|2.7193×|2.7386×|-0.332%|

加速比=T_Torch/T_自有；两个独立进程各自中位数先求均值再取比。双方使用同一真实B14/Q16输入、同一最终Cross K/V预计算缓存、warm完整单层CUDA Graph；原Torch通过静态slot适配器保留dense-history计算。每臂600次预热、24轮交错计时；每进程7527次完整层调用，检查输入/缓存/RNG不变。外部性能baseline只有原Torch；previous和R1仅作内部回归控制。

两进程三KV均小幅同向：最终相对此前阶段版下降约0.33%–0.36%，不同进程的单项变化范围0.089%–0.623%。幅度小，尚未确立实际应用中的显著收益；不能外推30层、chunk或机器人E2E。比同轮R1改善约0.71%–1.15%，不把跨轮旧数字混入本轮比值。准入和各KV检查无外部GPU进程，SM计时前后1410MHz，不改时钟。

[原始计时1](r2/retry/TIMING-1.json)、[计时2](r2/retry/TIMING-2.json)、[CSV](WHOLE-LAYER-TIMING.csv)、[汇总](GPU-SUMMARY.json)。本轮无新增NCU或逐阶段性能profile。

## OBSERVED：实际融合与依赖

- `whole_layer.cu:299`：每CTA按原静态分配完成FFN1 tile，GELU仍在写回执行；每28个生产tile组成一组，发布ACT可见性。112个生产tile不变。
- `:362`：两行块×两K7168段形成四组，每组24个FFN2任务。CTA完成自己的生产任务后领取已就绪消费者，不等待整阶段FFN1结束。
- `:411`：两个K段可以并行MMA；第二段只在commit前等同一行块第一段全部提交，FP32部分和仍保存在本CTA shared，原BF16/bias/gate/residual顺序不变。
- `:389`：全部96个消费者已有owner后，空闲CTA进入最终grid屏障，避免继续轮询。108常驻CTA中最多48个第二段CTA等待，其余CTA可推进第一段；仅在当前固定布局下核验。

使用device-scope acquire/release计数与每个写线程的device fence发布跨CTA数据；这些同步有成本。ACT及第一段F2仍在全局缓冲，没有宣称消除全部中间落地、减少HBM字节或形成专职warp流水。FFN局部grid同步由3次减为1次，整层warm29→27；数学任务3400/3472/3400不变，FFN仍112生产+96消费者。

## OBSERVED：数值、机制与资源

最终R2：9真实签名81调用，36结构/条件变体144调用，输出及23个共有中间量逐位一致；普通/debug、重复结果、输入、有效KV、RNG和合法B2缓存转移通过。四组生产计数均28、领取与完成计数均24。R1同规模有限门也通过；两套结果均保留。这不证明全部在线输入或机器人质量。

三档KV的debug记录均显示组0/1/2在最后一组FFN1发布前开始消费，且至少一个第二段MMA在第一段组全部提交前开始。此处仅为debug下的执行顺序证据，不将debug时间当正式阶段延迟。三档中最后生产组和最后消费完成组都为组3。

源码依赖仍保留“最后一组FFN1→该组FFN2”的串行链，解释了为什么提前完成部分任务并不必然大幅缩短整层；但这不是对全部剩余开销的定量归因。当前未拆分轮询、fence、设备调用栈和GEMM成本。

资源仍108CTA×256线程、1CTA/SM、100096B shared、255寄存器/线程。静态local报告由此前944B变为R1的1152B、最终1144B；包含设备调用/局部存储的报告，不能直接等同实测动态spill。正常增量状态16 int=64B；仅debug多12 int64=96B。

[9签名及顺序证据](r2/retry/SIMPLE-GATE-RESULT.json)、[36变体](r2/retry/STRUCTURAL-RESULT.json)、[R1源码](CANDIDATE-R1.cu)、[R2源码](r2/CANDIDATE-R2.cu)、[R2差异](r2/CHANGE.patch)、[阶段表](STAGE-MAP.md)。

## FAILED / UNKNOWN / RECOMMENDATION

R1曾将第二段的计算也等待第一段提交，同轮略慢。R2移除这项额外依赖并停止无用轮询；两个改动一起实施，不能将它们的收益分别量化。

R2首验证因build-lib输出目录仍指向父目录，在0次完整层调用时StopIteration退出。新R2产物已移到正确位置；R1从原对象文件重新链接后SHA256与首次计时身份完全一致，再进入新retry目录核验。原FAILED/日志保留，未重编数学代码或重置截止。见[产物修复](r2/ARTIFACT-REPAIR.json)、[原失败](r2/STATUS.json)、[retry终态](r2/retry/STATUS.json)。

下一步建议先量化最后组FFN1→FFN2的关键路径及发布/等待开销，再判断是否值得把FFN2段内消费做得更早；本轮不继续扩大调度搜索，不声称已完成整Transformer的MegaMoE式流水。更细分的归约流水或改变tile需新有限核验。

## 终态与预算

自有14个进程身份于2026-10-22T18:10:40.314298+08:00全部核验退出，GPU计算进程列表为空。GPU worker生命周期136.105秒；CPU构建293.356秒；包含准备/编译/等待/清理的保守窗口896.121秒，累计9844.678/36000秒，余26155.322秒。CUDA active未单独测量。原截止18:40:44保留。远端新增证据/构建约56.63MiB，小于512MiB上限。

[资源停止报告](RESOURCE-AND-STOP-REPORT.json)、[清理](FINAL-CLEANUP.json)、[远端采用](DEPLOYMENT-BAAI.json)、[本地采用](DEPLOYMENT-LOCAL.json)。四项科学权限及production_admitted全部false。
