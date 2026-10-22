# 单因素02：attention路由编译期专化

第一项内层forceinline无稳定收益，已恢复。第二项只把现有attention_dual的runtime cross改成模板bool，保留noinline dispatcher、原主循环/参数语义、tile与所有同步。希望缩短两路各自描述符/索引的活跃范围；不预设降低local或提高性能。

触发证据仍为当前整层动态local、静态callee存取与独立attention慢。沿用9签名逐位＋短中长五臂两进程，CPU≤1200秒，GPU≤720秒。最终若无收益恢复固定11；这是本段最后新实现候选。09:00与累计10h不变。
