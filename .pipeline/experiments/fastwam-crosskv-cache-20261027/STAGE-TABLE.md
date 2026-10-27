# C3/C4收口：chunk CrossKV缓存完整阶段表
评估并保留：显式缓存研究入口；生产仍原Torch、cross_cached默认false。下表阶段数字沿用本轮C2测量，C3只重测完整policy，未重测各阶段。layer15同轮独占5090，10次去首留9。倍率=Torch eager阶段/自有缓存阶段，>1较快；逐阶段隔离测量不能相加替代整层；两方对称Graph仅用于隔离device成本，直接API与正式compile单列。
|阶段|计算|任务量B1/B7|warp|B1倍率|B7倍率|
|---|---|---|---|---:|---:|
|1|调制表|24/168块×256|8向量|0.653×|1.072×|
|2|LN1＋调制|32/224行|4归约＋8写回|1.235×|1.115×|
|3|Self Q/K/V投影|72/288输出tile|1T4M3E|3.775×|1.943×|
|4|Q RMS＋RoPE|32/224行|8各归约一行|0.723×|0.891×|
|5|K RMS＋RoPE|32/224行|8各归约一行|0.717×|0.879×|
|6|KV拼接＋mask|1560/10920 KV块＋17/119 mask块|8向量|0.633×|0.421×|
|7|Self attention|24/168 head|8计算attention|1.162×|1.204×|
|8|Self输出投影|B1:128 partial＋128合并块; B7:32输出tile|1T4M3E；B1合并8|0.789×|0.764×|
|9|Self门控残差|128/896块|8向量|0.854×|0.898×|
|10|Cross LN|32/224行|4归约＋8写回|0.928×|0.697×|
|11|Cross Q投影|24/96输出tile|1T4M3E|1.342×|1.230×|
|12|Cross K/V投影|移出warm，预计算另计|原Torch预计算|预计算，warm为0|预计算，warm为0|
|13|Cross Q RMS|32/224行|8各归约一行|1.081×|1.300×|
|14|Cross K RMS|移出warm，预计算另计|原Torch预计算|预计算，warm为0|预计算，warm为0|
|15|Cross attention|24/168 head|8计算attention|1.178×|1.164×|
|16|Cross输出投影|B1:128 partial＋128合并块; B7:32输出tile|1T4M3E；B1合并8|0.726×|0.500×|
|17|Cross残差|128/896块|8向量|0.971×|0.945×|
|18|FFN LN＋调制|32/224行|4归约＋8写回|1.177×|1.086×|
|19|FFN1＋GELU|32/128输出tile|1T4M3E；E做GELU|1.034×|0.910×|
|20|FFN2|B1:128 partial＋128合并块; B7:32输出tile|1T4M3E；B1合并8|0.703×|0.643×|
|21|FFN门控残差|128/896块|8向量|1.025×|0.877×|

B1整层cached native 207.733 µs；相对同cache Torch eager 1.407×；相对同cache compile 0.605×。cache构建另计82.610 µs/层（不含text_embedding）。

B7整层cached native 463.804 µs；相对同cache Torch eager 0.925×；相对同cache compile 0.459×。cache构建另计113.860 µs/层（不含text_embedding）。
