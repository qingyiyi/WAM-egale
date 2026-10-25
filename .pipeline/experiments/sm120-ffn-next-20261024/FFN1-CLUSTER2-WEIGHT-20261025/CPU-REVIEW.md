# 槽与进度审查

OBSERVED（源码）：每cluster两个rank的tiles/steps相同。双槽ready按floor(step/2)翻转。
rank0 EMPTY=4个MMA读完+1个rank1 DSM完成ACK；rank1 EMPTY=4个MMA读完。
rank1先arm ready 24576B、发A 8192B并发目的credit，rank0 wait本地TMA完成与credit后发B 16384B。
rank1 ready完成才ACK源EMPTY，因而覆盖源不会和DSM读冲突。rank1自己回填前等本地MMA。
初装两槽，再处理step、回填step+2；无每K cluster.sync。入/出cluster.sync保护远程barrier寿命。
MMA沿48×K64全FP32归约；输出顺序/bias/BF16/GELU、FFN2七段和grid同步不变。
独立L只做同cluster+同任务映射+起止cluster同步，仍两次global TMA。

UNKNOWN：DSM等待净成本、实际L2/DRAM流量，待测；逻辑B加载减半不等于HBM减半。
CPU生成脚本首次因中文冒号匹配失败，0GPU调用，已修复文本定位；未产生候选运行或改写旧证据。
