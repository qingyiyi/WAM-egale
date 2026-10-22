# 已拒绝LTO候选逐阶段测量

OBSERVED：两个独立进程、3个真实B14 KV输入、28阶段。共有输入/输出/中间量、有效KV及RNG有限逐位核验通过；每进程3756次完整调用和125772次stage调用。原始JSON保留。

下表均为同一原版前缀的输入，两进程中位数的均值，单位µs。self_attention对照取guarded强路径，其余取原版。自有独立stage沿用整层108CTA/100096B shared资源，不是自由资源最优standalone。

|KV|阶段|原版/强对照|自有独立|自有整层嵌入|
|---|---|---:|---:|---:|
|short|table|30.720|32.768|28.672|
|short|norm1|13.312|19.456|14.848|
|short|modulate1|7.168|12.288|7.936|
|short|self_Q|43.008|49.152|49.152|
|short|self_K|38.656|46.080|41.472|
|short|self_V|38.912|45.056|41.472|
|short|self_Q_RMS|9.216|13.312|10.240|
|short|self_K_RMS|8.192|12.288|9.216|
|short|self_Q_RoPE|6.144|15.360|13.312|
|short|self_K_RoPE|6.144|14.336|11.776|
|short|self_attention|30.720|56.320|58.368|
|short|self_out|38.912|46.080|45.056|
|short|residual1|9.216|13.312|9.728|
|short|norm2|18.432|21.504|21.504|
|short|cross_Q|39.936|47.104|42.240|
|short|cross_K|553.984|566.272|772.096|
|short|cross_V|552.960|566.528|768.000|
|short|cross_Q_RMS|9.216|13.312|12.288|
|short|cross_K_RMS|69.632|118.528|141.312|
|short|cross_attention|99.840|101.888|103.936|
|short|cross_out|38.912|45.824|40.960|
|short|residual2|5.632|11.264|8.192|
|short|norm3|12.288|17.408|18.432|
|short|modulate3|8.192|13.568|10.240|
|short|FFN1|165.888|180.736|179.200|
|short|GELU|15.360|31.744|27.136|
|short|FFN2|116.736|125.184|124.928|
|short|residual3|9.216|13.312|9.728|
|middle|table|30.720|32.256|28.672|
|middle|norm1|13.312|19.456|14.336|
|middle|modulate1|7.168|12.288|7.168|
|middle|self_Q|43.008|48.896|49.152|
|middle|self_K|38.912|46.080|41.984|
|middle|self_V|38.400|45.056|40.960|
|middle|self_Q_RMS|9.216|13.312|10.240|
|middle|self_K_RMS|8.960|13.312|9.728|
|middle|self_Q_RoPE|6.144|15.360|13.312|
|middle|self_K_RoPE|6.144|14.336|11.264|
|middle|self_attention|156.672|215.552|270.336|
|middle|self_out|37.888|45.568|43.008|
|middle|residual1|9.216|12.544|9.472|
|middle|norm2|18.432|21.504|21.504|
|middle|cross_Q|39.936|47.616|42.496|
|middle|cross_K|553.984|565.248|772.096|
|middle|cross_V|552.960|566.016|768.000|
|middle|cross_Q_RMS|9.216|13.056|12.288|
|middle|cross_K_RMS|69.376|116.992|141.312|
|middle|cross_attention|99.328|102.400|104.192|
|middle|cross_out|38.912|44.800|40.960|
|middle|residual2|6.144|11.264|8.192|
|middle|norm3|12.288|17.408|18.432|
|middle|modulate3|8.192|13.312|10.240|
|middle|FFN1|166.400|181.504|178.944|
|middle|GELU|15.360|30.720|26.624|
|middle|FFN2|116.736|125.952|124.928|
|middle|residual3|9.216|12.544|9.216|
|long|table|30.464|32.768|28.672|
|long|norm1|13.312|19.456|14.848|
|long|modulate1|7.168|12.288|7.168|
|long|self_Q|43.008|49.152|49.920|
|long|self_K|38.400|46.080|41.216|
|long|self_V|38.656|45.312|41.984|
|long|self_Q_RMS|9.216|13.312|9.984|
|long|self_K_RMS|8.192|12.288|9.728|
|long|self_Q_RoPE|6.144|15.360|12.544|
|long|self_K_RoPE|6.144|14.336|11.776|
|long|self_attention|169.472|235.520|282.624|
|long|self_out|38.912|44.288|41.984|
|long|residual1|9.216|13.312|9.728|
|long|norm2|18.432|21.504|21.504|
|long|cross_Q|39.936|47.104|41.984|
|long|cross_K|554.496|565.760|772.096|
|long|cross_V|552.960|566.528|767.744|
|long|cross_Q_RMS|9.216|13.312|12.288|
|long|cross_K_RMS|69.632|118.016|145.664|
|long|cross_attention|99.328|102.912|103.680|
|long|cross_out|38.912|44.288|40.960|
|long|residual2|6.144|12.288|8.192|
|long|norm3|12.288|17.408|18.432|
|long|modulate3|8.192|13.312|9.984|
|long|FFN1|165.888|180.992|178.688|
|long|GELU|15.360|31.744|26.880|
|long|FFN2|116.736|126.208|125.952|
|long|residual3|9.216|13.312|9.216|

计时边界：独立stage为外部GPU marker区间，包含kernel边界；整层为阶段内部marker，包含原stage内部同步/分派/device调用。两者相减不是纯全局同步等待。各阶段中位数不能直接相加代替完整层计时。

插桩完整层相对活动二进制开销范围：0.232%–0.664%。插桩寄存器/local资源与非插桩略有不同，整层性能只引用03b-lto-target-fix的非插桩TIMING。

HYPOTHESIS：剩余差距含阶段几何/固定资源/函数组织；Cross-K RMS与GELU的独立stage也明显偏慢，不支持将全部差距归因于跨阶段全局等待。resource-envelope-control的资源干预支持“固定整层资源限制是部分原因”，但降低shared或增加CTA并非普遍改善。

UNKNOWN：此LTO负候选的动态local流量与SM stall未测；原selected11已有独立NCU，不移用其counter到LTO；静态localSizeBytes不能换算成实测spill字节。

RECOMMENDATION：本LTO候选已因整层稳定退化而拒绝；用本分解解释退化，不再追加新优化候选。保留单层限制和强B14基线，不用旧慢dense路径宣布完全对齐；不开始跨阶段ready-DAG。
