# T0：当前完整block阶段表

时间：2026-10-24T23:52:45.660155+08:00；评估变体：当前A的源码/历史证据；无新性能测量；实际保留：SELF-QKV-TMA-4MMA。
设备RTX5090/SM120，GPU-e5e7821f-9633-6cc8-d06c-d0d0df4c9256；三份真实B14/Q16/block0，KV288/1152/2160；warm CrossKV；eager/no Graph。
A固定SOURCE-MANIFEST.json与A-source/、独立A.so；当前历史资源170CTA×256线程、1CTA/SM、shared100352B、255reg、静态local784B；动态spill UNKNOWN。
源码SHA256：328dd69c0a20c57e81c19e8218c1a5e83125744baaa40d21c27194ff34a335f8；二进制SHA256：4adeeb937ad66f8b9cae57c93196a59c36730d2b76ce03a9cacd0d272ddebc35。

|阶段|形状/tile|CTA任务/owner|warp|ACT/partial/acc存储|同步|ready/free/提交条件|
|---|---|---|---|---|---|---|
|1. 布局准备|历史slot表；CTA0扫描|owner0；其余等待|8W|global索引/计数|grid1|layout完成|
|2. LN1＋调制1|224×3072；224行|全grid分行|W3–6归约；8W搬运|寄存器/shared→global M1|CTA归约＋grid1|完整行归约|
|3. Self QKV|3×96个64×128；K3072=9×320+192|共享288任务；owner0..169|W0 TMA；W1–3后处理；W4–7 MMA|双输入shared；acc寄存器→shared→global|grid2；每tile显式CTA5；mbarrier|输入ready/empty；结果full/empty|
|4. Self Q RMS＋RoPE|112双行tile；每行3072|owner0..111|两组各4W|global Q→shared/寄存器→global|CTA归约＋grid1|完整行后旋转|
|5. Self K RMS＋RoPE|112双行tile；每行3072|owner0..111|两组各4W|global K→shared/寄存器→global|CTA归约＋grid1|完整行后旋转|
|6. Self Attention/私有KV寻址|短/中/长192/336/192逻辑tile；K64循环5/18/34|96/168/96双组任务|两组各4W MMA|共享历史只读；私有K/V分开；acc寄存器|FA内部同步＋grid1|有效slot/mask/候选边界|
|7. Self输出投影|96个64×128；48×K64|96任务；动态owner|W0 TMA；W1–3后处理；W4–7 MMA|双输入shared；acc→shared→global|grid2；每tile显式CTA5；mbarrier|输入/结果槽ready/free|
|8. 残差1|336×2048元素|336静态任务|8W向量|global→寄存器→global R1|grid1|输出投影完成|
|9. LN2|224×3072；224行|全grid分行|W3–6归约；8W搬运|寄存器/shared→global N2|CTA归约＋grid1|完整行|
|10. Cross Q投影|96个64×128；48×K64|96任务；动态owner|W0 TMA；W1–3后处理；W4–7 MMA|双输入shared；acc→shared→global|grid2；每tile显式CTA5；mbarrier|输入/结果槽ready/free|
|11. Cross Q RMS|112双行tile|owner0..111|两组各4W|global Q→shared/寄存器→global|CTA归约＋grid1|完整行|
|12. Cross Attention|336逻辑tile/168双组；Q16/K512/D128|168双组任务|两组各4W MMA|缓存Cross K/V只读；acc寄存器|FA内部同步＋grid1|按相同条件组取K/V|
|13. Cross输出投影|96个64×128；48×K64|96任务；动态owner|W0 TMA；W1–3后处理；W4–7 MMA|双输入shared；acc→shared→global|grid2；每tile显式CTA5；mbarrier|输入/结果槽ready/free|
|14. 残差2|336×2048元素|336静态任务|8W向量|global→寄存器→global R2|grid1|Cross输出投影完成|
|15. LN3＋调制3|224×3072；224行|全grid分行|W3–6归约；8W搬运|寄存器/shared→global M3|CTA归约＋grid1|完整行|
|16. FFN1＋GELU＋FFN2＋门控残差|FFN1 108×128×256＋12×32×256；FFN2 96×7=672任务，M64/N128/K2048|生产owner0..119；消费owner0..169|FFN1主8W/尾4W MMA；FFN2 W3–6 MMA；8W提交|ACT global；P寄存器/shared；R0..5 global；OUT global|局部ready＋tile有序提交；出口grid1|同行两组生产完成；acquire s→release s+1|
|17. 退出|无数学tile|全部170CTA|8W|无新增状态|grid1|所有输出/状态完成|

warm Cross K/V投影与K RMS：NOT_EXECUTED；层外预计算不计入以上17行，不删除其数学工作。
FFN粒度：FFN1 hidden tile256，主M128/K_step32、尾M32/K_step64；ACT以四个生产组发布；FFN2数值K2048，内部K_step64，输出N128。FFN1重算1×。主循环shared与FP32 result复用同一union、不能同时保存两者；段内acc在寄存器；七段递推见NUMERIC-CONTRACT.md。

|阶段|A嵌入整层μs：短/中/长|原Torch对应μs：短/中/长|Torch/A：短/中/长|状态/run|
|---|---:|---:|---:|---|
|1. 布局准备|5.792/5.728/5.824|NA|NA|HISTORICAL_ONLY；10/24 final-comparison|
|2. LN1＋调制1|11.056/10.944/11.072|45.532/45.890/46.418|4.118×/4.193×/4.192×|HISTORICAL_ONLY；10/24 final-comparison|
|3. Self QKV|93.024/93.120/93.184|123.666/123.618/123.610|1.329×/1.328×/1.327×|HISTORICAL_ONLY；10/24 final-comparison|
|4. Self Q RMS＋RoPE|11.472/11.568/11.664|37.581/36.922/37.369|3.276×/3.192×/3.204×|HISTORICAL_ONLY；10/24 final-comparison|
|5. Self K RMS＋RoPE|10.976/10.976/11.136|36.957/36.954/37.230|3.367×/3.367×/3.343×|HISTORICAL_ONLY；10/24 final-comparison|
|6. Self Attention/私有KV寻址|38.208/102.976/183.872|127.988/592.026/1098.717|3.350×/5.749×/5.975×|HISTORICAL_ONLY；10/24 final-comparison|
|7. Self输出投影|48.432/48.832/48.944|41.731/41.717/41.702|0.862×/0.854×/0.852×|HISTORICAL_ONLY；10/24 final-comparison|
|8. 残差1|6.192/6.352/6.416|25.097/25.115/25.421|4.053×/3.954×/3.962×|HISTORICAL_ONLY；10/24 final-comparison|
|9. LN2|9.824/10.288/9.728|27.554/27.614/27.787|2.805×/2.684×/2.856×|HISTORICAL_ONLY；10/24 final-comparison|
|10. Cross Q投影|47.488/47.488/47.744|41.665/41.647/41.629|0.877×/0.877×/0.872×|HISTORICAL_ONLY；10/24 final-comparison|
|11. Cross Q RMS|6.048/6.192/6.624|9.886/9.744/10.013|1.635×/1.574×/1.512×|HISTORICAL_ONLY；10/24 final-comparison|
|12. Cross Attention|37.696/38.000/38.496|58.195/58.199/58.160|1.544×/1.532×/1.511×|HISTORICAL_ONLY；10/24 final-comparison|
|13. Cross输出投影|48.384/48.352/48.224|41.708/41.716/41.758|0.862×/0.863×/0.866×|HISTORICAL_ONLY；10/24 final-comparison|
|14. 残差2|3.280/3.264/3.488|5.523/5.533/5.495|1.684×/1.695×/1.575×|HISTORICAL_ONLY；10/24 final-comparison|
|15. LN3＋调制3|11.616/11.712/11.520|45.434/45.598/46.140|3.911×/3.893×/4.005×|HISTORICAL_ONLY；10/24 final-comparison|
|16. FFN1＋GELU＋FFN2＋门控残差|289.696/291.792/292.880|253.932/255.506/254.590|0.877×/0.876×/0.869×|HISTORICAL_ONLY；10/24 final-comparison|
|17. 退出|1.440/1.440/1.408|NA|NA|HISTORICAL_ONLY；10/24 final-comparison|

以上阶段均为旧A同轮profile区间及原Torch独立eager区域，不是本轮新测，也不能相加代替整层。新测正常完整block结果单列。

|KV|历史原Torch/A整层|历史Guarded/A整层|本轮A/候选|本轮正确性/决定|
|---|---:|---:|---|---|
|288|1.2514×|1.1127×|NOT_MEASURED|T0仅源码核对，保留A|
|1152|1.7546×|1.0845×|NOT_MEASURED|T0仅源码核对，保留A|
|2160|2.1902×|1.0453×|NOT_MEASURED|T0仅源码核对，保留A|

历史正常整层来自旧E任务两进程中的冻结A，不是E性能。本轮未改准入、容差、质量权限或production_admitted=false。
