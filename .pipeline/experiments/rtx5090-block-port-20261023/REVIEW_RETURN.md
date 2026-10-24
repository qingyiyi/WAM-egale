# RTX5090 单层迁移回传

OBSERVED：当前 L13＋FFN1 尾块两槽已在 SM120 编译并运行；三份真实 B14 输入完成两独立进程交错计时与阶段拆分。完整输出不满足原逐位契约，结论为 **COMPLETE_DIAGNOSTIC_NOT_QUALIFIED**。保留此迁移分支及全部失败，不替换原A100采用版本、不部署生产。

## 完整层观测

单卡RTX5090 32GB，170SM；Torch2.9.0+cu130、CUDA13.0.88、driver580.105.08。B14/Q16/D3072/H24/FFN14336。每个进程60次warmup，每臂/模式24个交错样本，每样本10次完整调用；表为两进程各自中位数再取中位。无插桩完整层二进制，单位ms。两进程不是置信区间。

原Torch是本地原始WanTransformerBlock类体，使用共同静态dense有效历史入口及共同warm最终Cross K/RMS/V缓存。没有guarded对照。eager是主口径，Graph是双方都捕获重放的补充，不把一方Graph与另一方eager比较。

| 总KV | 原Torch eager ms | 自有 eager ms | eager比值 | 双方Graph比值 | 输出相对L2 | 输出最大绝对差 |
|---:|---:|---:|---:|---:|---:|---:|
| 288 | 0.865721 | 0.709538 | 1.2201× | 1.1062× | 0.9387% | 2.5000 |
| 1152 | 1.341868 | 0.782188 | 1.7155× | 1.6024× | 0.7519% | 2.0000 |
| 2160 | 1.852038 | 0.865190 | 2.1406× | 2.0361× | 0.6759% | 2.0000 |

以上比值均为诊断观测，不是精度合格后的优化结果，也不是30层、chunk、任务E2E或机器人加速。Cross冷构建不计入warm时延，本轮未测其成本。

## 精度与差异定位

OBSERVED：三份真实输入的LN1与调制1逐位一致，首次共有中间量差异位于Self Q/K/V projection。Q投影相对L2约0.39%–0.40%，K/V也不一致。最终输出不同元素比例约61%–65%，不能仅凭相对L2小于1%认定可接受。所有有效KV、输入、CPU/CUDA RNG保持原值；正常/调试及正常/阶段插桩二进制输出在三fixture逐位相同。只覆盖这三个输入，B2及其余线上形状未测。

OBSERVED：旧compact-input预计算先将B14上下文压成两组，再运行Linear/RMS，在5090产生与原B14不同的缓存值。r1的prepared Torch自身也未逐位通过；原r1数据全部保留，不能用作最终口径。最终测量改为先读取原B14计算出的K/K-RMS/V，再验证七候选同CFG输出完全相同并去重保存。修正后缓存版Torch与canonical原Torch三个完整层逐位一致；自有输出仍未通过。

OBSERVED：现有self_qkv8_tile保留K1536×2以及中间BF16舍入。按同样两段构造的FP32诊断比全长FP32诊断更接近自有输出，但仍不逐位相同；不能称已经定位所有算术差异。当前Torch BF16允许reduced-precision reduction，其当前GEMM也不同于全长FP32参考。没有更改这个后端设置来制造通过。原A100 gold与5090 canonical Torch也不同；跨硬件不能继承旧逐位证书。

HYPOTHESIS：当前cuBLAS/SM120所选的累加、分段及舍入路径不同于先前为A100复现的路径，是首投影偏差的重要候选解释。尚未完成当前vendor kernel逐指令归因；后续RMS/attention/FFN的独立数值贡献UNKNOWN。不是已证明“只改一个BF16分段就可全部对齐”。

## 阶段与资源

完整27阶段表见[STAGE-TABLE.md](STAGE-TABLE.md)，r1快照独立保留。170CTA×8warp，shared100096B=97.75KiB，255寄存器/线程，静态local784B/线程，1CTA/SM。与99KiB/CTA上限相容；没有改成TMA、WGMMA或UMMA。

阶段比值显示Self Attention收益最大；它包含原Torch历史读取/拼接与attention路径，不能把整个差额都归于MMA或这次迁移的新增贡献。FFN1/GELU/FFN2/残差3联合段约0.736–0.740×，即自有慢约35%–36%；Cross Q RMS约0.64–0.68×，但绝对成本较小。三个剩余投影约0.92–0.96×。阶段自有区间包含同步等待，Torch为独立Graph，各边界不同，不作纯计算根因结论或可加性拆解。

阶段插桩相对正常二进制延迟扰动短/中/长约+0.23%/+0.27%/+1.09%，故接近1%的阶段差别不作强结论。未取得GPU硬件计数器，没有实测HBM/L2/occupancy/动态spill；源码静态资源不能代替这些指标。

## 这轮改了什么

1. 在独立复制的optimized入口将构建目标设为SM120；保留原A100入口与全部数学实现。
2. 修正CUTLASS 3.5.1 host adapter对CUDA13.0的版本判断；补齐隔离CUDA包nvvm/lib64和libnvvm.so布局。编译失败4次原样保存。
3. 移除本轮代码生成器内隐式SHA，按用户要求不执行hash；文件身份使用路径、尺寸、构建日志、归档和本机/远端源码逐字节核对。
4. 增加真实完整层迁移测量、独立阶段插桩二进制、首数值分歧诊断以及canonical输出去重缓存测量入口。未做tile/warp搜索、完整DAG、训练或仿真。

## 推荐下一步

先固定5090当前Torch的投影数值契约，单独对齐Self Q/K/V的分段、bias与BF16舍入，再传递核对剩余层。warm缓存采用原B14结果去重，而非未经核验的compact-input重算。达到有限严格数值门后重测完整层；之后优先处理FFN联合段，而不是先增加跨阶段调度。

5090现已能做后续研究基准，是否以其结果写正式表仍取决于数值/状态门和可确认的无争用窗口。本轮仅监测到自有CUDA进程，未取得宿主级独占证明。利用率异常结束时显示0%，原因未定位；没有为此重启或改驱动。

固定本机源码入口：
`workspace/算子研究/wam_ops/optimized/csrc/whole_layer.cu`

固定二进制：
`build/normal/wam_whole_fa2_cuda.cpython-312-x86_64-linux-gnu.so`

当前结果：`MEASUREMENT-SUMMARY.json`、`results/canonical-process1/RESULT.json`、`results/canonical-process2/RESULT.json`、`results/canonical-stage-data/RESULT.json`、`results/numeric-diagnostic-data/RESULT.json`。

所有四项科学权限保持false，production_admitted=false；原FAILED、retention、default、投机代码及tile保护未改。
