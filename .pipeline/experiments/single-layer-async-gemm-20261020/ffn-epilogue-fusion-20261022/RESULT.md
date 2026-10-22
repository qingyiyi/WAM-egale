# FFN写回融合：精度通过，当前未提速

2026-10-22完成。按用户授权，直接在optimized的B14路径实现FFN1+GELU、FFN2+门控残差3。两机研究入口采用融合版以供后续流水研究；这不是性能改进版本，原源码/二进制保留在before及上一轮证据目录。baseline、投机侧、原tile及四项科学权限不变。

## OBSERVED：只与原Torch比较

|KV|融合前自有相对原Torch|融合后自有相对原Torch|融合后相对自身旧版延迟变化|
|---|---:|---:|---:|
|288|1.1773×|1.1646×|+1.09%|
|1152|1.7085×|1.6893×|+1.14%|
|2160|2.7673×|2.7445×|+0.83%|

加速比=T_Torch/T_自有。两个独立进程的各自中位数先求均值再取比值；两进程均同方向小幅退化。仅原Torch作为外部性能baseline；旧自有为融合回归控制，没有guarded测试臂。

远端A100-SXM4-40GB GPU0，同真实B14/Q16输入，双方预计算最终Cross K/V，warm完整单层CUDA Graph边界；Torch使用既有静态slot适配器与原dense-history计算。每臂600次预热、24轮交错顺序；每进程5646次完整层调用。准入和各KV检查无外部GPU进程，测量前后SM1410MHz；不改时钟。不是动态原模型forward、30层或机器人E2E。

原始耗时仅作证据：[CSV](WHOLE-LAYER-TIMING.csv)、[进程1](TIMING-1.json)、[进程2](TIMING-2.json)、[汇总](GPU-SUMMARY.json)。本轮没有新逐阶段profile，不能把上一版本阶段时间贴到融合阶段，也不能将GELU独立阶段消失表述为无限加速。

## OBSERVED：代码与数值

- `whole_layer.cu:118`：原GELU公式抽为同一个inline函数，BF16输入转FP32后计算，再转BF16。
- `:293`：FFN1输出tile完成完整K归约、bias和BF16舍入后，直接执行GELU并写ACT；普通路径不分配/写F1，debug保留F1供审计。
- `:231`：FFN2保留两段K7168及第一段BF16落地；第二段按原加法次序完成后先舍入BF16，再计算原gate和残差并写OUT。普通路径只保留部分和F2，不再写最终F2供独立残差读取；debug仍保留最终F2。
- `:841`：仅candidate且M224走融合，删除该路径独立GELU与残差3调用；B2/action与B2/video仍走原阶段路径。

9种真实签名81次、36种结构/条件/调制变体144次完整层核验通过。输出及23个共有中间量逐位一致；普通/debug、重复结果、输入、有效KV、RNG及合法B2缓存转移通过。新增变体包含FP32 temb、单token广播及其组合。仅有限真实/真实派生输入，不能证明全部在线输入等价或提升机器人质量权限。

[9签名](SIMPLE-GATE-RESULT.json)、[36变体](STRUCTURAL-RESULT.json)、[源码差异](CHANGE.patch)、[源码身份](SOURCE-MANIFEST.json)、[远端采用](DEPLOYMENT-BAAI.json)、[本地采用](DEPLOYMENT-LOCAL.json)。本机仅采用已验证二进制，未另做本机GPU计时。

## OBSERVED：分块与资源

|项目|融合前|融合后|
|---|---|---|
|独立GELU任务|1568|0，计算并入112个FFN1输出tile|
|独立残差3任务|336|0，计算并入FFN2最终写回|
|FFN2计算任务|96个K分段、48个输出tile|不变|
|warm短/中/长逻辑任务|5304/5376/5304|3400/3472/3400|
|warm grid同步|31|29|
|CTA/线程/每SM驻留CTA|108/256/1|108/256/1|
|每CTA shared|100096B|100096B|
|每线程寄存器|255|255|
|静态local报告|936B|944B|

任务数与grid同步为源码核查口径，本轮未增加逐tile计数插桩。GELU/残差数学工作没有消失。逻辑全局张量访问最多减少F1读写12.25MiB和最终F2读写2.625MiB；部分和F2及ACT仍落地。这不是实测HBM流量；静态local增加8B也不能直接当成动态spill根因。

## HYPOTHESIS / UNKNOWN / RECOMMENDATION

已确定：当前FFN1的MMA和本tile GELU写回仍依次执行，尚无专职warp/双缓冲重叠；FFN2两分段及其同步保留。融合减少独立阶段，但增加了GEMM epilogue工作。

推测：FFN1尾波、GELU延长关键tile写回、编译寄存器/指令布局等可能抵消中间访问收益。尚未以本版本profile/单项消融区分贡献，不能把约1%的退化归因于某一个因素。

下一步建议先分清两处融合的增量，再设计MMA/epilogue双缓冲与依赖就绪交接；不把当前版本包装成MegaMoE式重叠流水。后续只比较相对原Torch的组合子图及完整层加速比。

## 终态与资源

所有自有作业于北京时间17:37:30核验退出，GPU进程列表为空。GPU worker生命周期65.147秒；包括准备、编译、等待和清理的本轮保守记账598.644秒，累计8948.557/36000秒，余27051.443秒。CUDA active时间未单独测量。首CPU编译bp变量遮蔽失败保留，GPU调用为0；修复r2沿同一18:12:32截止执行，未重置预算。

[清理收据](FINAL-CLEANUP.json)、[完整预算与停止报告](RESOURCE-AND-STOP-REPORT.json)、[首失败](STATUS.json)、[修复后终态](STATUS-r2.json)。
