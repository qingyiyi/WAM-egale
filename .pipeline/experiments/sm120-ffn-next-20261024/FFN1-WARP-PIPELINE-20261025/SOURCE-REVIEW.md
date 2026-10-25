# T0 源码与进度核对

OBSERVED：保留S独立源码及S.so；当前实现是FFN1流水W，public loader仍指S。原FFN2 partition64与commit64源码逐字节相同，只有ACT readiness阈值按448个真实tile变为112/112/112/112。

输入槽各ready由TMA完成事务翻转；每次K64四个MMA warp在完成最后shared load后各一次arrive empty。每warp在最后两次MMA时只读寄存器，此时生产者可覆盖槽。48次K64后结果发布；三个epilogue warp全部完成shared读取/ACT写出后释放结果槽，下一tileMMA无须等待，只在覆盖result前等待。

所有CTA在generic->async fence后入口grid同步；init/drain有CTA同步，热循环无全CTA同步。每CTA各角色共享同样静态tile列表，不跨CTA等待；进入FFN2之前仍有全grid同步。

正确性重点：四MMA输出分配使用现有CutlassTile64的32×64 warp shape；完整K内每K16累加顺序不变，bias后BF16再GELU。input/result物理缓冲不alias。当前未获得W实测结果，不将源级可并行证明当实际重叠时长。

旧FFN2 INPUT/EPILOGUE/KREADY及C-OVERLAP负结果保留；本次作用于FFN1完整K结果并把120大tile换448小tile，不能声明是单因素warp数量实验。
