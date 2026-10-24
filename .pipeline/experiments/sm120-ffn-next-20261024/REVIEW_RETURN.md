# SM120 FFN接续：NO_GO，保留当前A

OBSERVED：T0/T1/T2完成一个PAIR2048x2设计。相邻两段同CTA，组内BF16状态shared交接，七段原FP32 partial与BF16 running逐段有限逐位通过；原捕获中间量、输出、输入/有效KV/RNG和新384-owner schema通过。没有改容差、GELU、bias/残差、模型、权重或ACT。

正常整层两独立进程在所有三KV均慢6.25%–7.89%；同轮FFN profile候选/A约1.18–1.19倍。结论NO_GO，当前SELF-QKV-TMA-4MMA源码和二进制保持不变。不是数值失败，不晋级。

## 可复核事实

- A/候选正常资源完全同为170CTA/1CTA每SM/shared100352B/reg255/static-local784B。不能把退化归因为扩大shared或静态资源导致occupancy下降。
- 72KiB compute union加16KiB BF16状态可放进现有96KiB TileShared，没有压小M或重复FFN1。候选确实保留ACT和七次舍入，仅将672段任务分为384 owner组。
- T1前序等待摊分仅0.20–0.22μs/任务，MMA与输入/结果交接包络25.8μs，提交转换1.9μs、发布0.43μs。不是纯Tensor Core时间，也不是全GPU关键路径比例。
- 最大可省7.875MiB是源码逻辑global往返，增加同量shared交接；没有DRAM/L2实测，不写HBM节省。

HYPOTHESIS：任务变粗导致split-K负载均衡/尾波变差，与退化方向一致。但新提交逻辑和任务归属同时改变，没有逐tile硬件trace或受控分离，不能把全部退化直接归因该因素。既有静态资源不变排除了“单纯多占shared”的解释，不能排除动态spill/cache等影响。

UNKNOWN：DRAM/L2分项、真实动态spill、critical CTA任务分布、精确Tensor Core忙碌率。软件包络不是NCU；没有为已明确负收益再编译新的诊断候选。

## 用户五个问题

1. 当前最合理的保留配置是已验证A：FFN1主M128/N256/K32，尾M32/N256/K64；FFN2 M64/N128、七个K2048、内部K64。它是在已测方案中更合适，不是全局最优证明；更细ACT ready粒度尚未验证。
2. 通过global ACT让24个输出列tile共享一次FFN1，重算1×；代价是ACT存储/读取与ready发布。此前M16/DSM无重算仍慢，本轮不重跑。
3. 保留FFN1原cp.async/MMA→bias/BF16→GELU/BF16→ACT global→release ready；FFN2原四warp MMA→shared FP32 result→八warp逐段合并/转换/写回→fence/release。六投影的专职TMA路径与本轮FFN是不同子图，未改。
4. 七段逐次舍入完整保留。s0=B(P0)，s1..5=B(Ps+FP32(Rprev))，s6=B((P6+FP32(R5))+b2)，最终再按原门控残差顺序舍入。每段P都独立清零，不把上一段R作为MMA初值。三真实fixture的全部七段原始P/R已保存比较；不外推全部输入。
5. 保留global ACT和当前全局partial方案，因为本轮省partial逻辑量未转化收益，驻留旧方案也无收益。没有证据支持现在强行删除ACT或新调warp比例。

## 决定与下一步

停止PAIR2048x2，不做第二设计或tile/warp搜索。建议下一轮优先解释FFN2约25.8μs的MMA/输入交接包络，或单独讨论更细ACT ready边界；两者不在本轮自动展开。允许答案是当前global ACT版本暂时最合适。

完整表STAGE-TABLE.md；机器结果RESULTS.json；原始正常数据T2-r1/results/measure{1,2}-data/RESULT.json；同轮局部paired-local-comparison/data/RESULT.json；T1数值原件T1-r1/data/numeric-*.pt；候选数值T2-r1/results/measure1-data/numeric-candidate-*.pt；独立A.so、A-source、T2-r1/CHANGE.patch及FINAL-SOURCE-MANIFEST.json固定源码/二进制身份。

本机与远端科研根不是Git checkout，不能伪造HEAD；原始来源为已发布3eb0026705e603d6268731bb26b34261acca696c及现场字节一致的采用记录，当前固定文件身份优先。公开main镜像提交另存publication收据，不以远端main覆盖判断。

engineering_quality_screen=false；independent_test_complete=false；noninferiority_established=false；formal_performance_permission=false；production_admitted=false。这里只是有限单层真实重放，不是在线verifier或机器人E2E。
