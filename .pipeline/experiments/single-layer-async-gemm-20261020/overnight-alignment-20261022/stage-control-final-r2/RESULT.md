# 最终单层逐阶段测量

OBSERVED：两个独立进程、3个真实B14 KV输入、28阶段。共有输入/输出/中间量、有效KV及RNG有限逐位核验通过；每进程3756次完整调用和125772次stage调用。原始JSON保留。

下表均为同一原版前缀的输入，两进程中位数的均值，单位µs。self_attention对照取guarded强路径，其余取原版。自有独立stage沿用整层108CTA/100096B shared资源，不是自由资源最优standalone。

|KV|阶段|原版/强对照|自有独立|自有整层嵌入|
|---|---|---:|---:|---:|
|short|table|30.720|32.256|28.672|
|short|norm1|13.312|19.456|15.616|
|short|modulate1|7.168|12.800|7.168|
|short|self_Q|43.008|57.344|54.016|
|short|self_K|38.912|54.272|44.800|
|short|self_V|38.912|53.248|45.056|
|short|self_Q_RMS|9.216|13.312|11.264|
|short|self_K_RMS|8.192|13.312|9.216|
|short|self_Q_RoPE|6.144|15.360|13.312|
|short|self_K_RoPE|6.144|14.336|12.288|
|short|self_attention|30.720|66.560|60.416|
|short|self_out|38.912|52.992|51.712|
|short|residual1|9.216|13.312|10.240|
|short|norm2|18.432|21.504|22.528|
|short|cross_Q|39.936|55.552|46.336|
|short|cross_K|553.984|579.584|574.720|
|short|cross_V|552.960|581.120|569.344|
|short|cross_Q_RMS|8.960|13.312|9.984|
|short|cross_K_RMS|69.632|120.320|117.760|
|short|cross_attention|100.096|121.856|114.944|
|short|cross_out|38.912|52.224|46.336|
|short|residual2|6.144|11.264|9.216|
|short|norm3|12.288|18.432|18.944|
|short|modulate3|8.192|14.336|10.240|
|short|FFN1|165.888|203.264|199.168|
|short|GELU|15.360|31.488|29.184|
|short|FFN2|116.224|144.896|138.240|
|short|residual3|9.216|13.312|10.752|
|middle|table|30.720|31.744|28.672|
|middle|norm1|13.312|19.456|15.360|
|middle|modulate1|7.168|12.288|7.168|
|middle|self_Q|43.008|56.576|54.272|
|middle|self_K|38.912|53.248|45.056|
|middle|self_V|38.656|53.248|45.056|
|middle|self_Q_RMS|9.216|13.312|11.264|
|middle|self_K_RMS|8.704|13.312|9.216|
|middle|self_Q_RoPE|6.144|15.360|13.312|
|middle|self_K_RoPE|6.144|14.336|12.288|
|middle|self_attention|156.672|235.520|230.912|
|middle|self_out|37.888|52.224|48.128|
|middle|residual1|9.216|13.312|10.240|
|middle|norm2|17.920|21.504|22.528|
|middle|cross_Q|39.936|55.808|46.080|
|middle|cross_K|553.984|579.072|574.208|
|middle|cross_V|552.960|581.120|569.856|
|middle|cross_Q_RMS|9.216|13.312|11.008|
|middle|cross_K_RMS|69.632|116.736|116.480|
|middle|cross_attention|99.328|120.320|115.456|
|middle|cross_out|39.424|52.224|46.592|
|middle|residual2|6.144|11.264|9.216|
|middle|norm3|12.288|17.920|18.432|
|middle|modulate3|8.192|13.824|10.240|
|middle|FFN1|166.656|202.752|198.912|
|middle|GELU|15.360|30.720|29.696|
|middle|FFN2|116.736|144.896|138.240|
|middle|residual3|9.216|13.056|10.240|
|long|table|30.464|31.744|28.672|
|long|norm1|13.312|19.456|15.360|
|long|modulate1|7.168|12.288|7.168|
|long|self_Q|43.008|56.832|54.016|
|long|self_K|38.912|53.248|45.056|
|long|self_V|38.144|53.248|45.056|
|long|self_Q_RMS|9.216|13.312|11.264|
|long|self_K_RMS|8.192|13.312|9.472|
|long|self_Q_RoPE|6.144|15.360|13.312|
|long|self_K_RoPE|6.144|14.336|12.288|
|long|self_attention|168.960|252.416|247.808|
|long|self_out|38.912|52.480|47.616|
|long|residual1|9.216|13.312|10.240|
|long|norm2|18.432|21.504|22.528|
|long|cross_Q|39.936|55.296|46.080|
|long|cross_K|553.984|580.352|572.928|
|long|cross_V|552.960|581.376|570.624|
|long|cross_Q_RMS|9.216|13.312|10.496|
|long|cross_K_RMS|69.632|117.760|115.968|
|long|cross_attention|99.328|120.832|115.712|
|long|cross_out|38.912|52.224|46.592|
|long|residual2|6.144|11.264|9.216|
|long|norm3|12.288|17.408|18.432|
|long|modulate3|8.192|13.312|10.240|
|long|FFN1|166.144|202.240|198.656|
|long|GELU|15.360|30.720|30.464|
|long|FFN2|116.224|144.896|138.240|
|long|residual3|9.216|12.800|10.240|

计时边界：独立stage为外部GPU marker区间，包含kernel边界；整层为阶段内部marker，包含原stage内部同步/分派/device调用。两者相减不是纯全局同步等待。各阶段中位数不能直接相加代替完整层计时。

插桩完整层相对活动二进制开销范围：0.401%–0.917%。插桩寄存器/local资源与非插桩略有不同，正式整层性能只引用final-whole-comparison。

HYPOTHESIS：剩余差距含阶段几何/固定资源/函数组织；Cross-K RMS与GELU的独立stage也明显偏慢，不支持将全部差距归因于跨阶段全局等待。resource-envelope-control的资源干预支持“固定整层资源限制是部分原因”，但降低shared或增加CTA并非普遍改善。

UNKNOWN：当前版本动态local流量/实际achieved occupancy/SM stall归因等待本机NCU；静态localSizeBytes不能换算成实测spill字节。

RECOMMENDATION：先以当前最终版本NCU检验内存和函数边界，再决定下一轮最小改动。保留单层限制和强B14基线，不用旧慢dense路径宣布完全对齐；不开始跨阶段ready-DAG。
