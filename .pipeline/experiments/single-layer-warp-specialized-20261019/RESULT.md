# 整层一个persistent kernel：结构完成，精度与速度未验收

固定Long block0权重，9份真实完整block输入，B2/B14 action及B2 video×短中长KV。CUDA源码df8b386…；本机88保护项、远端86已部署项不变。远端未迁入state/rules，两Missing导致原始88项检查为false，原样保留，不创建缺失保护文件。0完整30层、0机器人episode。

9条GPU trace各观察到一次整层计算launch；队列tile恰好一次、输入/RNG/参数flags/地址通过。self/cross attention、全部投影、FFN、Norm/RoPE/残差和KV管理都在内，8 warp按调度/加载/计算/写回分工。当前仍有global中间量及粗阶段屏障，无跨阶段细流水。

| 输入 | Torch ms | 冻结native ms | 新整层 ms | 相对native延迟 | hidden输出相对L2 |
|---|---:|---:|---:|---:|---:|
|short-B2-action|1.9681|1.4740|7.5525|5.12×|0.002373|
|middle-B2-action|2.0603|1.5816|8.5417|5.40×|0.003599|
|long-B2-action|2.1939|1.7229|9.6911|5.62×|0.003173|
|short-B14-action|3.3060|2.5334|28.5809|11.28×|0.005597|
|middle-B14-action|3.6915|2.4494|26.8012|10.94×|0.003438|
|long-B14-action|4.9126|2.8856|30.5347|10.58×|0.004274|
|short-B2-video|2.2149|1.8913|13.5982|7.19×|0.001307|
|middle-B2-video|2.3158|1.9988|16.9585|8.48×|0.001698|
|long-B2-video|2.4576|2.1427|20.9731|9.79×|0.001372|

每格两独立进程、各5次交错样本，共10样本中位；270原始样本、分进程数据见CSV。profile与计时分离。不锁时钟，GPU1另卡任务及host条件留在原资源记录。这是未准入实现的成本诊断，不能称有效加速或论文正式性能。强cached/shared-prefix/Graph尚未比较；当前相对较弱native已明显更慢。

原Torch输出/历史gold与冻结native有限对照通过；新实现9/9最终hidden输出逐位FAIL，相对L2约0.13%–0.56%，max abs见JSON。这不是物理动作误差或任务失败率。只读9有效KV通过；合法写序列18中9数值失败，metadata均过。新单kernel不得继承原接受路径、完整模型或闭环质量证书。

相同输入Q/FFN2 Linear控制证实GEMM已有数值差异。公共cuBLASLt API重建读到：B2 FFN2 splitK7/inplace，B14 Q与FFN2 splitK2/inplace；新WMMA是单段FP32累加。配置与舍入契约不同，不能仅由metadata指定唯一根因；旧M26四公式失败不改、不重复无限扫参。

实际函数172 registers/thread、59616B shared、LOCAL0，occupancy API理论1 CTA/SM、108 CTA。这是资源上限而非实测achieved occupancy。首版较小64×64×32 GEMM tile、同步加载、粗屏障和global workspace都有额外成本；精确阶段占比等待独立设备内计时，未将推断写成观察。

远端三GPU job共66.555秒、450完整block尝试、54Linear控制；全部root真实reap并释放context。总CUDA active秒UNKNOWN。首次本机错误Python依赖失败0block和前两编译失败均保留。

下一步：补一次已登记的阶段profile，仅在GEMM占比≥50%且9输入保持v1位相等时测试一个ldm40布局挑战者。当前主源码/DSO冻结，不改baseline/投机侧，不集成，不扩30层，不以放宽精度门换PASS。
