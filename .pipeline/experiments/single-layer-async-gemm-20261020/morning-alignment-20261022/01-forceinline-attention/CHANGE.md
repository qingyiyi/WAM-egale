# 单因素01：FA2内层函数强制内联

证据：最终版本真实B14中KV NCU的local读写243623424B；静态SASS可见FA2调用及local存取。动态计数不能定位到单个函数，内联是可证伪干预，不预设收益。

仅对两个B14 FA2 tile主循环加forceinline；保持已有noinline attention dispatcher、原数学和所有同步。9真实签名逐位后，三KV五臂两进程对比固定11。若退化/无稳定收益则恢复。CPU构建≤1200秒，GPU三阶段≤720秒，09:00截止，存储≤1GiB。
