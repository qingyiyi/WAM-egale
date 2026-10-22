# FFN2写回与函数调用开销：两项保留，独立FFN仍慢约9%

## OBSERVED

用户确认先FFN2写回，再设备函数保存恢复，并补测独立FFN。沿M39直接修改optimized，未改baseline/投机/保护文件。只有两项实现尝试，没有分块或参数网格搜索。

A：FFN2每线程两元素改四元素写回；循环32→16次，列偏移、bias和固定调制表读取移出行循环。保留(partial2+bias)+BF16(partial1)、最终BF16舍入、FP32门控残差的顺序。vector4和常量读取外提作为一项写回实现改动，未分别消融两者贡献。

B：在A上仅将action_ffn1_produce从noinline改成forceinline，合入action_ffn_pipeline；尾部helper及pipeline本身仍是设备函数。没有全部内联，分块、依赖、96个FFN2任务和数学顺序不变。

|B14 KV|A：原Torch/完整层|A延迟下降（相对尾部分摊版）|B：原Torch/完整层|B延迟下降（相对A）|当前独立FFN 原Torch/自有|
|---|---:|---:|---:|---:|---:|
|288|1.2083×|1.611%|1.2185×|1.020%|0.9182×|
|1152|1.7435×|1.349%|1.7605×|1.156%|0.9167×|
|2160|2.8299×|1.334%|2.8473×|0.811%|0.9161×|

完整层为远端A100-SXM4 GPU0、warm CUDA Graph、双方预计算最终Cross K/V、原Torch静态合法slot适配；准入和逐case采样未见其他compute进程，频率采样1410MHz，未改时钟。每项两个独立进程，三KV×150预热×24交错轮，每事件10次replay。每进程每档均改善。A与B各自有同轮内部回归控制，不把不同轮的改进百分比相加或乘成组合实测。外部性能baseline只有原Torch。

每项9真实签名、36结构控制，共各225次有限核验；23个共同中间量、最终输出、输入、有效KV/RNG、repeat/debug/普通路径通过。计时fixture与精度fixture路径逐项匹配。不是全部输入的数值定理，不自动继承投机路径/闭环质量。

## 写回和调用边界证据

A的debug最后生产到最后消费间隔约136→121us，生产就绪间隔基本不变；支持改动作用于FFN2后段，但并未用单阶段计时把vector4与外提分别归因。

B的独立producer设备函数从编译函数列表消失。A的producer/pipeline静态HMMA分别64/128，B合并后192，尾部helper32不变；这是二进制静态计数，不是执行FLOPs或利用率。producer/pipeline合计静态STL/LDL由254/245变为129/128；含尾部helper后两边均另加17/16。函数调用边界减少，但其余寄存器分配也可能改变，不能把全部延迟改善归给某一种访存。

每CTA仍256线程，108常驻CTA，shared100096B，每线程255寄存器。每线程静态localSizeBytes：尾部分摊版1224B→A1232B→B920B。B相对A下降25.32%；不表示动态local流量或HBM流量下降同样比例。本轮没有NCU动态计数，未改变驱动/权限。

## 独立FFN重放

选定B源码派生测量副本，只加FFN-only入口与真实M3/R2指针绑定，不加入逐tile时间戳。独立入口含计数初始化可见性屏障；整个kernel及资源配置与完整副本相同。选定版本完整层与副本完整层的差异每进程每档绝对值不超过0.24%，资源字段完全一致（local920B），有限输出/输入/KV/RNG检查通过。

原Torch独立FFN包含FFN1+bias+GELU、FFN2+bias、门控残差；gate来自原前置调制表，自有按需计算gate，沿现有边界计费。两个进程三输入，原Torch/自有为0.9182/0.9167/0.9161×，自有仍慢8.912%/9.084%/9.154%。约0.347–0.348ms对0.318–0.319ms。测量副本中FFN1/GELU/FFN2/输出亦逐位一致。

旧R2独立FFN慢23.4%–23.8%是先前诊断；本次约9%是当前各自对原Torch的测量，不是把旧数值按单项收益推算出来。不能宣称FFN已追平或将独立FFN比值称为整个Transformer/机器人速度。

## UNKNOWN / RECOMMENDATION

剩余约9%不能再直接归因于同步，也不能说只剩MMA吞吐。FFN生产、依赖流水、写回、调度和函数开销仍共同存在；当前重放没有将自有FFN1/FFN2完全分开对齐。下一步先做当前版本两个GEMM阶段的同输入/同边界计算与搬运对照，再只优化主要差距；暂不继续减小tile或全部内联。

A、B均保留为可追溯证据；两机研究入口采用B。warm逻辑任务仍3408/3480/3408、27grid；ACT仍落全局。四项科学权限及production_admitted保持false，机器人队列不启动。本机只同步源码/DSO，未运行本机GPU重测。

[结果CSV](RESULTS.csv)、[汇总](SUMMARY.json)、[A源码差异](a-vector4/CHANGE.patch)、[B单行差异](b-inline-producer/CHANGE.patch)、[A原始计时](a-vector4/TIMING-1.json)、[B原始计时](b-inline-producer/TIMING-1.json)、[独立FFN](replay/SUMMARY.json)、[SASS](b-inline-producer/SASS-SUMMARY.json)、[阶段表](STAGE-MAP.md)、[资源清理](RESOURCE-AND-STOP-REPORT.json)。

本轮GPU worker生命周期149.253秒，CPU编译442.124秒；保守含准备/等待/清理1042.947秒，累计12338.592/36000秒。无本轮失败/timeout，16个owned身份退出，旧截止/FAILED保留。新截止2026-10-22T22:06:20.408234+08:00未重置。
