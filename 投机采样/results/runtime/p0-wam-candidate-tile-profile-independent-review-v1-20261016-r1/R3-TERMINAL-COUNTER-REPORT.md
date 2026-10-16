真实三launch机制profile终态独审通过：原CSV恰好一个units行和ID0/1/2，模板顺序native、G1、G2正确，三者各11个NCU pass；producer COUNTERS与独立解析原CSV完全一致，238来源当前字节与真实两pilot前置保持。

|arm|profile时间us|DRAM Mbyte|L2请求Mbyte|CTA|regs/thread|shared Kbyte/block|achieved occupancy|
|---|---:|---:|---:|---:|---:|---:|---:|
|native_shared_BM128|407.232|70.192512|508.791392|336|255|66.560|10.467%|
|research_G1_BM64|296.480|67.076352|477.612928|336|168|50.176|17.862%|
|research_G2_BM64|171.296|66.196608|320.893280|192|168|50.176|11.146%|

G1→G2：DRAM字节下降1.3116%，L2请求字节下降32.8131%，CTA下降42.8571%；achieved occupancy变化-6.7164个百分点（下降）。这支持减少CTA与重复L2请求的有限机制解释，不能称HBM减半或更高occupancy。

NCU日志明确警告缓存及频率未控制，cache-control/clock-control均none、每launch11pass。两次warm和实际输入数值门保持，但profile时间不进入正式性能，也不能据此推断完整30层、chunk或e2e加速。Mbyte按1000000字节换算，L2请求不是唯一KV存储。

实际总预算墙钟27.685秒，GPU阶段及CPU导出11.128秒，证据2406991B；profile及export四个具名身份均已退出，保存末次NVML为空，27样本无外来GPU。三实际输出及全部有效输入/KV/CFG/slots/RNG/版本门记录通过。审阅者仅CPU，无GPU/SSH、未重跑。
