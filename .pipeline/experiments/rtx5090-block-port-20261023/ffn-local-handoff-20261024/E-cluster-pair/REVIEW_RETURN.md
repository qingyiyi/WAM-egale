# E双CTA DSM：实际可运行，但不采用

OBSERVED：170CTA按cluster2运行85组，rank0生产64KiB ACT后双方各消费12个N128输出tile。三KV真实27中间量/输出、输入/有效KV/RNG及336×7提交有限逐位通过；全kernel cluster launch实际成功。两个进程整层均退化，短/中/长为A的3.12329/2.93193/2.74515倍；同轮FFN局部约6倍。保留A字节不变。

代码同时改变M16分块、静态驻留任务、权重复用及整个grid的cluster放置；不能把退化归因DSM一项，不能由本结果断言所有双CTA方案无效。新schema是循环完成载荷/实际提交计数，非硬件请求计数。四权限与production_admitted=false。

完整数值/原始计时在results，STAGE-TABLE及snapshots保存，固定whole_layer.cu/CHANGE.patch/SOURCE-ENTRY.json；旧编译失败及其他负结果不改。
