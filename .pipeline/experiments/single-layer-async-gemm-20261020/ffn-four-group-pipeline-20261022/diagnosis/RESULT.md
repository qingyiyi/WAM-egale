# FFN瓶颈定位：尾部生产、FFN2计算/写回与函数保存恢复

2026-10-22；生产R2源码和DSO未修改。用户本轮只批准先定位FFN，未实施新调度或tile优化。

## OBSERVED：同输入原Torch完整FFN子图

|KV|原Torch / 自有FFN|自有延迟增加|
|---|---:|---:|
|288|0.8080×|23.77%|
|1152|0.8095×|23.54%|
|2160|0.8102×|23.42%|

边界：真实B14、M224/D3072/F14336；从M3调制后输入至最终门控残差。双方同一M3/R2、权重和temb；Torch使用原FFN模块及原FP32残差表达式，gate为前段已生成值；自有保留按需gate计算及原BF16部分和。独立CUDA Graph，各150次warm、24轮交错，每次event包含10次重放；两个独立进程。仅远端A100-SXM4的正常计时进入比值。两次进程分别约0.8072–0.8106×。全FFN图与逐组件图因缓存/执行边界不同，不相加伪造整体。

诊断副本关闭记录相对生产整层差−0.12%至+0.43%；开启时间戳相对同副本关闭差−0.32%至+0.06%。这说明本轮整层扰动较小，不证明逐tile调度完全无扰动。诊断版255regs、1160B静态local；生产255regs、1144B。插桩内部时间线和独立图events分列，不将二者直接计算阶段正式加速比。

[原始进程1](replay-fix/PROFILE-1.json)、[进程2](replay-fix/PROFILE-2.json)、[汇总](SUMMARY.json)、[CSV](FFN-COMPARISON.csv)。实际输出、FFN中间参照、输入/有效KV/RNG均在三真实输入通过有限逐位检查；不外推所有输入、机器人质量或正式性能权限。

## OBSERVED：关键时间线

两进程×三KV×12次，72条完整FFN时间线一致：最后生产和最后消费组均3。

- 112个FFN1 tile，108CTA静态分配；CTA0–3再做tile108–111，全部属于组3。组0/1/2生产约131–138us结束；组3约243–245us结束。
- 最后组仍须在28生产tile全部就绪后派24个FFN2半段，故保留末尾串行链。最后生产结束到最后消费结束约127–128us。
- 最后一个消费者中，partition约86–87us（含MMA、迭代器准备及shared部分和写回），其中MMA warp区间约85–86us；最终epilogue约32us；提交依赖等待约1us；发布约2us。其余是调度/出口等间隙。
- 因此最终尾部约68%是FFN2计算段、25%是最终写回，提交等待不足约1%。不能把这个比例当整个FFN的时间占比。
- 组1第二段提交等待通常约3–6us；组3约1us。第一段写回约4–5us，第二段约31–39us，后者还做BF16部分和合并、bias、按需gate和残差。
- CTA累计时间中出口屏障约23.7%，队列/未归因约14.6%。前者主要记录先完成CTA等待最后任务，不能直接解释成屏障指令占整层23.7%、也不能推断去掉屏障就能节省23.7%。
- 平均约65个CTA处于计算/写回区间；这是时间线派生覆盖，不是NCU SM/Tensor Core利用率。

整层内FFN插桩span约375–377us，独立分支span约373–375us，差约0.5%–0.9%。这次没有观察到FFN嵌入整层后的大幅新增退化；独立FFN自身已慢于Torch。FFN约占当前自有整层插桩边界的33%–39%，不是WAM/video/episode比例。

[完整分解](replay-fix/ANALYSIS.json)、[代表时间线PNG](FFN-TIMELINE.png)、[SVG](FFN-TIMELINE.svg)。代表样本按12次span中最接近中位数选择，非挑最好结果。globaltimer观测粒度约1us，亚微秒差异不作为根因。

## OBSERVED：本机NCU与当前生产汇编

远端NCU因ERR_NVGPUCTRPERM退出，未改权限；本机无计算进程准入后使用同诊断二进制、关闭时间戳、独立FFN分支补采。这里只作硬件计数，未混写远端速度。当前本机已有hlj新任务，启动时间晚于本次NCU结束，不作清理。

|计数|自有独立FFN|原Torch对应路径|
|---|---:|---:|
|local读取|27.537408 MB|0|
|local写入|28.147968 MB|0|
|FFN1 GEMM grid/block|自有生产阶段112逻辑tile、108常驻CTA×256线程|112 CTA×256线程|
|FFN2 GEMM grid/block|96半段任务、复用108CTA×256线程|96 CTA×128线程|
|Tensor Core active cycles|自有整个FFN混合kernel37.33%|FFN1 GEMM77.19%、FFN2 GEMM81.85%|

最后一行范围不等：自有包含GELU、调度、等待及残差，不能说自有MMA核心只有Torch一半效率。自有barrier stall42.04%同样是warp stall指标，不能直接当延迟可消除比例。local计数是LSU请求字节，不等于同量HBM流量。

当前生产DSO的SASS进一步显示：FFN1共119条STL、118条LDL静态指令，流水函数133条STL、125条LDL；主要连续块在函数进入/返回时保存恢复寄存器。已检查的首末HMMA之间无LDL/STL；不能将这些local访问笼统写成MMA主循环频繁溢出。生产FFN1编译报告472B spill stores/loads，流水实际clone496B；诊断副本476/504B，二者相近但不把诊断计数冒充生产路径精确计数。

FFN2最终写回的生产源码仍逐循环计算row/batch/token及temb地址，含通用除余与类型分支；汇编epilogue含对应倒数/整数指令。它是可检验的优化点，尚未用单因素改动证明其贡献，不能把31–39us全部归因到除法。

[NCU摘要](local-counters/COUNTER-SUMMARY.json)、[原始CSV](local-counters/RAW.csv)、[生产SASS分析](SASS-ANALYSIS.json)、[FFN1汇编](PRODUCTION-FFN1.sass)、[FFN2汇编](PRODUCTION-FFN2.sass)。二进制源SHA仍为`7b7c81383a726b22d953d76a51ff6fcde60dc5f31d0e7be4d9f31a8679edbd1a`。

## RECOMMENDATION / UNKNOWN

1. 优先处理FFN1最后4个tile使组3延后的路径。建议有限比较尾部工作分摊/合适的小tile，而非直接把全部tile放大；原Torch FFN1也有112CTA，故tile数本身不能作为唯一慢因。
2. 接着处理FFN2最终写回：固定B14形状下简化索引、复用gate地址、核对向量宽度；同时降低非内联设备调用的寄存器保存恢复开销。保留算术与舍入次序，先分别量化，不一口气合并多个改动。
3. 暂不优先改“第一段全部24输出完成才提交”的等待粒度；当前关键尾部该等待约1us，实测投入价值低于计算/写回。

未知：每项新改动能实际节省多少、local访问延迟贡献、MMA内具体stall分布，以及能否传到30层/闭环任务。上述是下一轮建议，本次没有实施或启动优化。

## 资源与失败

新窗口原截止2026-10-22T19:24:12.494633+08:00保留；自有GPU worker生命周期合计98.889秒，CPU编译147.173秒，准备/等待/清理保守计1039.686秒，累计10884.364/36000秒。CUDA active未单独测量。两端owned已退出；生产源码、DSO、baseline、投机、原tile及保护文件不改；四项科学权限false。

保留三类失败：宿主/容器PID误比导致0算子退出；捕获Graph未replay就比较的无效结果；远端NCU权限失败。前两项修复测量脚本后，同一二进制、同一截止在replay-fix完成。没有把失败改PASS，也没有新机器人任务。

[资源报告](RESOURCE-AND-STOP-REPORT.json)、[远端清理](REMOTE-CLEANUP.json)、[本机清理](LOCAL-CLEANUP.json)。
