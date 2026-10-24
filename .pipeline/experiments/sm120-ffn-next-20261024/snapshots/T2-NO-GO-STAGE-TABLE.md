# T2-NO-GO：实际保留A完整阶段表

时间2026-10-25T00:36:24.168952+08:00；评估PAIR2048x2；结论NO_GO；实际保留SELF-QKV-TMA-4MMA，原优化入口未改。
RTX5090/SM120 GPU-e5e7821f-9633-6cc8-d06c-d0d0df4c9256；B14/Q16/block0，KV288/1152/2160；warm CrossKV；eager/no Graph。
A源码/独立二进制与候选的完整身份在FINAL-SOURCE-MANIFEST.json；A的源码/DSO SHA256分别为328dd69c0a20c57e81c19e8218c1a5e83125744baaa40d21c27194ff34a335f8 / 4adeeb937ad66f8b9cae57c93196a59c36730d2b76ce03a9cacd0d272ddebc35。
正常A/候选同资源：170CTA×256线程，1CTA/SM，shared100352B、255reg、静态local784B；profile静态local792B。动态spill/DRAM/L2计数UNKNOWN。

|阶段|形状/tile|CTA任务/owner|warp|ACT/partial/acc存储|同步|ready/free/提交|
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

以上主结构描述实际保留A。被拒候选只改第16行：96输出tile×4组=384 owner任务，仍672段MMA；组内R0/R2/R4 shared，组间R1/R3/R5 global，七次舍入；group边界release 2/4/6/7。FFN1重算1×，ACT仍global。
FFN1主M128/N256/K_step32，尾M32/N256/K_step64；FFN2 M64/N128，数值K2048、内部K_step64。原ACT四生产组发布，候选不改。候选compute73728B+BF16 running16384B=90112B，小于现有TileShared98304B。
warm Cross K/V投影与K RMS为NOT_EXECUTED；层外预计算另列。

|阶段|A μs 短/中/长|原Torch μs 短/中/长|Torch/A|被拒PAIR μs 短/中/长|A/PAIR|状态/来源|
|---|---:|---:|---:|---:|---:|---|
|1. 布局准备|5.792/5.744/5.824|NA|NA|5.792/5.744/5.808|1.000×/1.000×/1.003×|CURRENT_MEASURED；paired-local-comparison|
|2. LN1＋调制1|11.008/11.072/11.136|44.784/45.124/45.155|4.068×/4.076×/4.055×|11.088/10.960/11.072|0.993×/1.010×/1.006×|CURRENT_MEASURED；paired-local-comparison|
|3. Self QKV|92.800/93.056/93.072|124.990/123.542/123.486|1.347×/1.328×/1.327×|94.528/95.040/94.864|0.982×/0.979×/0.981×|CURRENT_MEASURED；paired-local-comparison|
|4. Self Q RMS＋RoPE|11.520/11.504/11.632|53.998/36.979/36.878|4.687×/3.214×/3.170×|11.648/11.680/11.808|0.989×/0.985×/0.985×|CURRENT_MEASURED；paired-local-comparison|
|5. Self K RMS＋RoPE|10.992/11.040/11.040|40.883/36.970/36.878|3.719×/3.349×/3.340×|11.024/11.072/11.232|0.997×/0.997×/0.983×|CURRENT_MEASURED；paired-local-comparison|
|6. Self Attention/私有KV寻址|38.128/103.552/184.368|127.830/591.656/1097.284|3.353×/5.714×/5.952×|38.048/103.696/184.960|1.002×/0.999×/0.997×|CURRENT_MEASURED；paired-local-comparison|
|7. Self输出投影|48.368/48.896/48.912|41.719/41.682/41.698|0.863×/0.852×/0.853×|48.256/48.864/48.864|1.002×/1.001×/1.001×|CURRENT_MEASURED；paired-local-comparison|
|8. 残差1|6.192/6.272/6.352|24.329/24.504/24.647|3.929×/3.907×/3.880×|6.240/6.224/6.400|0.992×/1.008×/0.993×|CURRENT_MEASURED；paired-local-comparison|
|9. LN2|9.776/10.240/9.760|27.216/27.213/27.438|2.784×/2.658×/2.811×|9.744/10.240/9.696|1.003×/1.000×/1.007×|CURRENT_MEASURED；paired-local-comparison|
|10. Cross Q投影|47.360/47.616/47.664|41.616/41.629/41.616|0.879×/0.874×/0.873×|47.408/47.552/47.648|0.999×/1.001×/1.000×|CURRENT_MEASURED；paired-local-comparison|
|11. Cross Q RMS|6.048/6.240/6.080|10.115/9.832/10.016|1.672×/1.576×/1.647×|6.000/6.240/6.048|1.008×/1.000×/1.005×|CURRENT_MEASURED；paired-local-comparison|
|12. Cross Attention|37.712/38.176/38.224|58.197/58.183/58.228|1.543×/1.524×/1.523×|37.792/38.288/38.640|0.998×/0.997×/0.989×|CURRENT_MEASURED；paired-local-comparison|
|13. Cross输出投影|48.288/48.288/48.208|41.524/41.665/41.690|0.860×/0.863×/0.865×|48.384/48.192/48.224|0.998×/1.002×/1.000×|CURRENT_MEASURED；paired-local-comparison|
|14. 残差2|3.312/3.280/3.456|5.486/5.527/5.508|1.656×/1.685×/1.594×|3.296/3.296/3.616|1.005×/0.995×/0.956×|CURRENT_MEASURED；paired-local-comparison|
|15. LN3＋调制3|11.520/11.632/11.504|45.131/44.987/45.513|3.918×/3.868×/3.956×|11.472/11.552/11.472|1.004×/1.007×/1.003×|CURRENT_MEASURED；paired-local-comparison|
|16. FFN1＋GELU＋FFN2＋门控残差|289.632/292.960/291.872|255.132/255.479/254.965|0.881×/0.872×/0.874×|345.440/345.712/346.480|0.838×/0.847×/0.842×|CURRENT_MEASURED；paired-local-comparison|
|17. 退出|1.408/1.440/1.424|NA|NA|1.440/1.440/1.440|0.978×/1.000×/0.989×|CURRENT_MEASURED；paired-local-comparison|

阶段为两份独立profile DSO的嵌入区间，原Torch是同job独立eager数学区域；含同步/重叠，不能相加代替整层。完整block采用判定只用以下无插桩原始记录。

|进程/KV|A μs|PAIR μs|A/PAIR|配对比值P10–P90|PAIR退化|正确性/决定|
|---|---:|---:|---:|---|---:|---|
|1/short|693.059|747.738|0.9269×|0.9233–0.9313|+7.89%|有限逐位PASS；NO_GO，保留A|
|1/middle|760.912|819.814|0.9282×|0.9267–0.9328|+7.74%|有限逐位PASS；NO_GO，保留A|
|1/long|844.310|897.114|0.9411×|0.9337–0.9414|+6.25%|有限逐位PASS；NO_GO，保留A|
|2/short|692.864|747.238|0.9272×|0.9220–0.9304|+7.85%|有限逐位PASS；NO_GO，保留A|
|2/middle|761.872|816.410|0.9332×|0.9309–0.9356|+7.16%|有限逐位PASS；NO_GO，保留A|
|2/long|840.691|896.323|0.9379×|0.9348–0.9423|+6.62%|有限逐位PASS；NO_GO，保留A|

P10–P90为16个交错轮次的描述性波动区间，不是置信区间。全量样本保存在RESULT.json，未挑最快样本。

|KV|当前同轮原Torch/A|当前同轮Guarded/A|FFN A/PAIR（同轮profile）|
|---|---|---|---|
|short|1.2508×/1.2510×|1.1098×/1.1100×|0.8384×|
|middle|1.7606×/1.7584×|1.0894×/1.0840×|0.8474×|
|long|2.1912×/2.2007×|1.0469×/1.0474×|0.8424×|

有限正确性范围是三份真实block0输入；其他层/在线接受/完整模型/闭环质量、Graph replay和B2 fallback未扩验，不借用本结果放行。旧Graph与旧A100结果未混入。
