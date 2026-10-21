# 同一原语的资源配额控制

OBSERVED：当前10版本实际中KV输入，7阶段×两种前序×两进程，原输出/输入/RNG及有效KV有限一致。完整当前层的21个结构变体另验全部通过，见RESULT.json。每个device函数数学/编译代码相同，仅独立阶段launch的dynamic shared和CTA grid改变；结果不安装入整层。

|同原版前缀，µs|原版|整层shared/108CTA|最小shared/108CTA|最小shared/更多CTA|
|---|---:|---:|---:|---:|
|Norm1|13.312|19.456–20.480|20.480|14.336（324CTA）|
|CrossK RMS|69.632|115.712–118.784|124.928|94.208（216CTA）|
|GELU|15.360|31.744|32.768|23.552（864CTA）|
|调制表|30.720|28.160–33.792|24.576|20.992–24.576（432CTA）|
|调制1|7.168|13.312|13.312|14.336（540CTA）|
|Self Q RoPE|6.144|16.384|17.408|16.384（432CTA）|

全部原始样本及配置见RESULT-1/2.json、CONTROL.csv。余量并非都能靠更多CTA消除。这里的active数来自cudaOccupancyMaxActiveBlocksPerMultiprocessor，是可驻留上限，不能称NCU实测occupancy。

INFERENCE：固定整层100096B shared、108CTA确实限制部分原语，CrossK RMS和Norm1/GELU有直接干预证据；仅减shared且保持108CTA，并未统一改善。更换shared配额也可能影响L1/cache行为，但没有新NCU计数，不能把差异全部归为L1或HBM。RoPE与调制小阶段还有执行/同步等成本，不能用occupancy单因素解释。

UNKNOWN：单个persistent kernel的每阶段资源不能按上述独立launch自动切换；这些数不是已实现的整层收益。当前完整层仍255寄存器/线程且1CTA/SM，不可把独立stage较低寄存器与增加CTA直接套入完整层。旧NCU计数属于旧源，不混作本轮动态流量。

RECOMMENDATION：清理已静态分派阶段的冗余入口同步，并只做一次大RMS的两warp/行等归约树对照。其后按完整层收益选版，停止新增实现，完成同轮最终对照和证据包；跨阶段DAG仍未实现。
