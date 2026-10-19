# 10/20联合汇报：单层算子、四套LIBERO与迁移

证据汇总时间2026-10-20T03:23:44.569965+08:00。

整层单persistent kernel已经实际运行，warp分工也已实现；当前原型的原精度和速度验收失败，不能替换冻结baseline。600调用有限研究已收口，没有把结构完成写成系统收益。

|输入|冻结native ms|新整层ms|新/旧耗时|hidden相对L2|
|---|---:|---:|---:|---:|
|short-B2-action|1.4740|7.5525|5.12×|0.002373|
|short-B14-action|2.5334|28.5809|11.28×|0.005597|
|long-B14-action|2.8856|30.5347|10.58×|0.004274|

全部九签名、270原始计时样本、只读/写KV、RNG/参数/API与错误均保留。首轮B14 GEMM74.66%；唯一布局诊断下降6.5%–19.0%，但仅匹配失败v1，原门不变。profile/pad寄存器175/198、理论均1 CTA/SM。主要问题是成熟设备原语数值契约、同步单缓冲与粗阶段执行，不是少几个Python调用。[源码与问题](../single-layer-warp-specialized-20261019/IMPLEMENTATION-ISSUES.md)。

|套件|Target|TF|TF+PF2|
|---|---:|---:|---:|
|long|483/500（96.6%）|484/500（96.8%）|473/500（94.6%）|
|goal|463/500（92.6%）|464/500（92.8%）|467/500（93.4%）|
|spatial|472/500（94.4%）|466/500（93.2%）|464/500（92.8%）|
|object|497/500（99.4%）|498/500（99.6%）|499/500（99.8%）|

6000完整局不等于独立TEST；最大成功长度比八组合均>1.30，全部失败/800罚分保留，TF/PF2没有通过原工程质量门。PF2在Long/Spatial整体成功数更低，不推荐默认开启。首两S审计还发现第一次周期T之前的相机或before-KV差异，不能把其后差异全归于PF2。[质量门](QUALITY-RESULT.md)、[首两S路径](PAIRING-DIFFERENCES.md)。

新工程计时状态CLOSED_PARTIAL_PRE_REGISTERED_PANEL，330/360；Object首轮0episode实施失败不重跑，最大330。独立server重复只用于计时，Goal/Spatial/Long各臂最多30，Object各臂20；不是新独立质量样本。

|套件|T/C/P计时完整N；成功数|TF/PF2有效更新速度比|TF/PF2完整policy/chunk|TF/PF2全体暖episode|TF/PF2双方成功暖episode|
|---|---|---|---|---|---|
|long|30/30/30；30/27/27|1.625×/1.350×|1.363×/1.218×|1.107×/1.009×|1.344×/1.226×|
|goal|30/30/30；29/30/29|1.595×/1.339×|1.358×/1.216×|1.523×/1.231×|1.319×/1.162×|
|spatial|30/30/30；26/27/27|1.545×/1.331×|1.328×/1.213×|1.421×/1.321×|1.253×/1.170×|
|object|20/20/20；20/20/20|1.575×/1.341×|1.341×/1.216×|1.216×/1.099×|1.216×/1.099×|

全体完整attempt的暖episode速度比、双方成功N、原始逐局、三/两重复的均值与SD另列[完整计时表](ISOLATED-RESULT.md)，没有删除失败或成功长尾。有效更新=Σ原action round时间/Σ合法scheduler g；不是每条树的接受率。warm包含reset/RPC/cache/环境/审计、排除env.close和落盘，冷setup单列；不是无审计生产cold E2E。新whole kernel没有装入这些作业。[账本](COST-LEDGER.md)、[边界](TIME-BOUNDARY.md)。

RoboTwin：官方LingBot-VA权重确实存在；50唯一任务，官方适配每任务100个通过expert过滤的有效seed，一条件三臂全量15000局。其Q32/有效16动作通道、静态pool9792和三相机不同，需要新draft/normalizer/接口及质量验证；当前不适合直接复用LIBERO TF64。[官方权重](https://huggingface.co/robbyant/lingbot-va-posttrain-robotwin)、[完整迁移核查](../robotwin-migration-survey-20261019/RESULT.md)。没有启动新环境。

下一步：先处理小M Q、FFN2与attention的同输入设备数学契约；随后只验证一条SM80兼容的tile异步流水路线，双方同用成熟cache/shared-prefix/Graph。不继续小布局搜索，不扩30层或环境，不重训来掩盖系统问题。论文还缺少步数Target25/17、匹配MLP等强对照，不能宣称投机优于直接少步数；当前未启动这些作业。10/22做go/no-go；10/23冻结，10/28主表，11/01 18:00中文证据包。缺口不闭合则收缩主张。[时间安排](DATES-AND-PLAN-20261020.md)。

engineering_quality_screen=false；independent_test_complete=false；noninferiority_established=false；formal_performance_permission=false。baseline及依赖、投机源码和原tile/HW/保护文件不改。数据/权重不重复迁入，owned退出与资源硬限见[止损](STOP-REPORT.md)；原四文与短入口已归档后语义重建。GitHub实际提交号以发布收据为准。

终态资源核对：远端匹配身份的active owned PID 0、CUDA compute context 0；本机M34 active 0。12个phase guard均CLEAN；没有留新GPU任务排队。所有失败及原预算保留，不关闭服务器实例。
