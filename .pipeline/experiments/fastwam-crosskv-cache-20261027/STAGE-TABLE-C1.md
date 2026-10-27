# C1源码/编译完成，缓存版性能未测
评估：chunk内CrossKV；默认：未缓存严格版。两stage移出warm block，原始编号保留。
|阶段|计算|任务量B1/B7|warp|性能|
|---|---|---|---|---|
|1|调制表|24/168块×256|8向量|未测|
|2|LN1＋调制|32/224行|4归约＋8写回|未测|
|3|Self Q/K/V投影|72/288输出tile|1T4M3E|未测|
|4|Q RMS＋RoPE|32/224行|8各归约一行|未测|
|5|K RMS＋RoPE|32/224行|8各归约一行|未测|
|6|KV拼接＋mask|1560/10920 KV块＋17/119 mask块|8向量|未测|
|7|Self attention|24/168 head|8计算attention|未测|
|8|Self输出投影|B1:128 partial＋128合并块; B7:32输出tile|1T4M3E；B1合并8|未测|
|9|Self门控残差|128/896块|8向量|未测|
|10|Cross LN|32/224行|4归约＋8写回|未测|
|11|Cross Q投影|24/96输出tile|1T4M3E|未测|
|12|Cross K/V投影|移出warm，预计算另计|原Torch预计算|未测|
|13|Cross Q RMS|32/224行|8各归约一行|未测|
|14|Cross K RMS|移出warm，预计算另计|原Torch预计算|未测|
|15|Cross attention|24/168 head|8计算attention|未测|
|16|Cross输出投影|B1:128 partial＋128合并块; B7:32输出tile|1T4M3E；B1合并8|未测|
|17|Cross残差|128/896块|8向量|未测|
|18|FFN LN＋调制|32/224行|4归约＋8写回|未测|
|19|FFN1＋GELU|32/128输出tile|1T4M3E；E做GELU|未测|
|20|FFN2|B1:128 partial＋128合并块; B7:32输出tile|1T4M3E；B1合并8|未测|
|21|FFN门控残差|128/896块|8向量|未测|
