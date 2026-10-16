# 私有GPU进程归属监控：CPU修复与验证

2026-10-11T21:26:49+08:00（Asia/Shanghai）

## 当前结论

投机采样新GPU job资源门通过但加载阶段被私有PGID监控误停：两个torchrun rank各自新建session，已由父launcher日志确认，非外部抢占。26.05秒触发/28.40秒清理记录，0EGL/0固定chunk/0完整episode，成本未测。新CPU私有监控改按父子链+root PID启动tick/UID/argv认领，原Torch constructor/_popen AST真实子进程拓扑及26回归通过，同用户无关进程仍拒绝。1347冻结文件/11原实现/9target stat不变，GPU不重跑、不训练，非tile。

这是新增CPU-only prototype，不修改本次GPU失败job的guard/manifest/plan或11个实现；不能被旧manifest运行，也未复制到任何新GPU执行job。原工程221项CPU gate与本次新增26项CPU回归是两个不同测试集，不冒称统一247项GPU验证。

## 修复机制

- 不按同PGID/同用户/命令名字直接放行。新registry root先检查真实argv、UID与登记PGID，并pin PID/start_ticks身份防PID重用。
- 每个可见compute PID实时走/proc PPID祖先链到精确root身份；允许本任务rank另建session、子孙多级另建PGID。UID变化、PID复用、无关同UID进程、仅同PGID、循环或父子starttick倒置不放行。
- NVML与/proc退出竞态的已消失PID略过；活的孤儿/不可检查/非法NVML数据保守失败。只请求本任务STOP，无foreign signal函数。
- manifest额外版本2与自身/tests/plan SHA绑定，默认关闭。真实旧job manifest在任何GPU查询前拒绝该prototype。

## CPU实测

[26项日志](../../../review/OMISSIONS.md)：PASS（0.348秒）。23项基础与接口回归加原Torch拓扑/两个CPU watcher接口用例，共unittest报告26项，subTest不作为独立计数。

- 执行当前安装Torch的原SubprocessHandler.__init__与_popen AST，并启动三组真实CPU parent/child：前两组登记归本任务，第三组同UID/类似命令但未登记。每组child PPID指向对应parent，PGID与parent不同。旧PGID-only guard对前两组误报，新guard放行前两组并拒绝第三组。
- NUMA context使用明确CPU no-op；没有import Torch，没有CUDA初始化或GPU模型/环境进程。NVML字符串是合成文本，真实的是/proc拓扑，不拿它冒充GPU/NVML实测。
- watcher主循环的外部GPU查询及contract许可显式mock：模拟无关PID发本任务STOP、正常终态退出；旧manifest拒绝测试调用实际冻结contract但不调用GPU查询。
- 首次测试harness只提取_popen而缺原constructor-owned stdout字段失败，日志ownership-CPU-tests-attempt1.log保留；最终改为同时执行原constructor，未手填其字段。中间24-pass日志亦保留，最终26项全过。
- CPU parent/child均自然结束；原四个GPU launcher/rank PID不存在。1347冻结文件无变、target九项size/mtime无变、保护文件无变，缺失cycle.yaml保持缺失。

## 尚未验证

真正GPU启动/reset/draft-ready、两EGL初态、GPU cache恢复与G1、成本四格仍未过；新guard在真实GPU持续运行与任意并发短任务/强制超时孤儿清理也未得到此CPU测试证明。same-agent实现与测试，不是独立评审。采样式进程查询不证明采样间绝无未观测GPU工作。

## 下一步

按失败停：不恢复/修改旧job，不自动GPU重试或训练。CPU私有监控v2已验证；若用户另确认新job与0完整episode/2EGL/168固定chunk/90分钟预算，重新预留GPU窗口和冻结CPU gate后，真实启动→两EGL初态→fixture恢复→G1→四格GPU成本。

正式性能仍是GPU上的原target T与A_off+冻结verifier，CPU时钟配合CUDA同步测实际路径wall time、CPU比较器只复核GPU产出，均不等于CPU模型测速。
