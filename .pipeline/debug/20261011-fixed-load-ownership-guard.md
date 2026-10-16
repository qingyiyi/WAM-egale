# 固定负载GPU启动误停：进程归属debug

2026-10-11T21:26:49+08:00（Asia/Shanghai）；scope=speculative-sampling，非tile。

## 症状

CPU工程关通过后实际GPU加载约28秒即stop/deadline requested，timed_out=false，无EGL/chunk。

## 排序假设与验证

1. **私有PGID-only guard误报：已证实。** 误报两个PID与父torchrun日志发送SIGTERM的rank完全匹配；原Torch_popen新建session，实际CPU parent/child拓扑重现旧错、新规则通过。
2. 外部GPU任务抢占导致停止：未支持。本次STOP中两个PID都是本任务，资源门0MiB/无PID。后来21:20:42另一个用户任务是独立后续状态，不是此次失败原因。
3. reset/action_per_frame或draft mode仍错：本次未走到reset，不可判定GPU修复成败；221项CPU生命周期证据保留。
4. GPU安全限时/端口/RAM/磁盘/source gate问题：启动资源PASS且26秒停止、无timeout，具体触发来自STOP_REQUESTED；不支持。

## 根因与交付

我新增加的私有资源observer假设rank与launcher同进程组，11个model/成本实现未变。新增独立CPU归属规则prototype及26项回归，旧job全部保留、GPU不重试。强制超时孤儿清理与完整GPU仍需独立验证，不能以正常TERM清理替代所有清理场景。

[尝试报告](../research/speculative-sampling/FIXED-LOAD-COST-ATTEMPT-20261011-205205.md)；[CPU修复](../research/speculative-sampling/FIXED-LOAD-OWNERSHIP-CPU-FIX-20261011.md)；[实际拓扑](../../投机采样/results/runtime/p0-fixed-load-ownership-cpu-fix-20261011-211415/actual-child-topology-CPU.json)。

下一步：按失败停：不恢复/修改旧job，不自动GPU重试或训练。CPU私有监控v2已验证；若用户另确认新job与0完整episode/2EGL/168固定chunk/90分钟预算，重新预留GPU窗口和冻结CPU gate后，真实启动→两EGL初态→fixture恢复→G1→四格GPU成本。
