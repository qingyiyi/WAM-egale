# DSM分片FFN实测与采用决定

**NO_GO：保留原A，未将DSM接入模型。** 一个cluster8新设计及一次定位修订均完成，不再追加网格。

本轮在同一RTX5090、三档真实B14输入、warm CrossKV、eager/noGraph下比较。A独立二进制保留，Torch和guarded都重测；两个独立进程的无插桩整层结果用于采用判断。阶段和等待插桩仅作归因。

|KV|A ms范围|B ms范围|R1 ms范围|R1较A增加|Torch/R1范围|
|---|---:|---:|---:|---:|---:|
|short|0.6937–0.6944|1.1038–1.1048|0.9524–0.9525|37.1491–37.2989%|0.9091–0.9100|
|middle|0.7604–0.7615|1.2522–1.2534|1.0998–1.1020|44.6400–44.7297%|1.2149–1.2189|
|long|0.8411–0.8416|1.2556–1.2602|1.1038–1.1053|31.1618–31.4114%|1.6736–1.6744|

OBSERVED：正常路径不再生成完整global ACT。8CTA共同生产各自两个64×128 GELU片，每CTA消费者负责三个64×128输出，借DSM将片复制到本地shared后做MMA。28个cluster job覆盖四行块×七K2048段；448生产片、672分段输出。4生产warp与4消费warp并行，只有128线程组barrier、READY和cluster作业边界同步；外层仍阶段同步。

数值递推保持：每段P_s从0开始按原K16 MMA顺序计算；R0=BF16(P0)，s=1..5时R_s=BF16(P_s+float(R_{s-1}))；末段R6=BF16((P6+float(R5))+bias)。最终门控残差沿原边界；未改成七段FP32只舍入一次。

OBSERVED：三档真实输入的27个中间量/最终输出逐位相同，七段FP32 partial及保存的running trace均相同；默认/custom stream、有效KV、输入和RNG有限核验通过。不是完整模型、闭环或所有输入的保证。

OBSERVED：R1仅把消费者下一W2片TMA提前到当前MMA之前。FFN插桩span从B约0.679ms降至0.525–0.527ms，但原Torch完整FFN约0.255ms；完整层仍比保留A慢31%–45%。

OBSERVED：插桩每CTA作业组leader中位等待中，consumer TMA约97.6–98.2μs→12.8–13.4μs，consumer总区间约337–340→261–263μs；DSM READY等待仍约72–73μs，MMA+组同步约86→107μs。区间跨warp/CTA重叠，不能相加为整层，也不是NCU stall百分比。

OBSERVED：资源为152驻留CTA、256线程/CTA、100352B shared、255register/thread、784B静态local/thread。A为170CTA；本轮cross-attention的168双组任务在152CTA中要分波，这是未改attention本体也变慢的调度解释。未采NCU动态访存，不能把静态local字段等同实际spill流量，不能声称HBM减半。

HYPOTHESIS：固定cluster分组、三输出累加器与并发生产消费的资源竞争共同抵消ACT去global收益；当前证据没有分离每项贡献。

UNKNOWN：动态DSM/L2/DRAM流量、不同cluster/tile/consumer布局的最优点。禁止将本轮负结果推为DSM普遍无效。

RECOMMENDATION：本轮到此止损。下一步如继续，优先在A上定位FFN2计算/交接成本；只有先证明cluster对非FFN阶段无净损和消费者资源可控，才值得另立DSM方案。本轮不再改容量、扩warp或重写DAG。

固定入口：`R1/whole_layer.cu:1007`生产；`:1056`消费；`:1076`DSM读取；`:1122`cluster调度；`:1919`cluster launch。补丁`R1/CHANGE.patch`。A是`../A.so`及`A-whole_layer.cu`；本机选中optimized源码未改。

原始证据：`MEASURE1.json`/`MEASURE2.json`、`R1/MEASURE1.json`/`MEASURE2.json`；`results`与`R1/results`保存默认/custom精度和原始numeric trace，`COST-SUMMARY.json`与`R1/COST-SUMMARY.json`保存局部计时摘要。完整17阶段见`STAGE-TABLE.md`。

四项科学权限仍false；投机侧/原tile/全局默认/旧FAILED未改。资源闭合及同步收据另见`RESOURCE-AND-STOP-REPORT.json`。
