# 资源中断后的有界接续

原第1轮在短KV完成22项数值/状态检查，24次full30、30次block，GPU生命周期37.297340944秒。进入中KV前发现其他GPU进程，原PARTIAL/未完成第二进程保留，旧轮计时不作正式比较。

沿同一10:14:26起点/13:14:26截止、1200秒/150次full30/500次block帽接续，不增加预算。新的两进程均核对original/baseline/optimized全部六block与六full30点、合法write1/2；第1进程三臂三轮计时，第2进程original/baseline三轮计时。故baseline两独立进程，optimized只有第一进程性能与两进程数值，不伪称其两进程完整性能重复。预计与原轮合计150次full30、168次block。前瞻登记后才启动；原3臂两进程性能计划仍PARTIAL，四false与数值门不改。
