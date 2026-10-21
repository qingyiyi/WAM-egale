---
kind: decision
name: 允许本机NsightCompute做算子诊断
level: constraint
scope: wam-research-20261016/M39
status: active
sources:
  - 用户2026-10-21明确允许算子放到本机用Nsight Compute profile以辅助慢因定位
supersedes: []
---
# 允许在本机进行Nsight Compute诊断

本轮当前算子慢因定位可以使用本机GPU和Nsight Compute；先核占用、权限和实际源/输入身份。在本机不可用时可继续已授权远端GPU0，避免为相同证据重复投入。不同机器的计数/时钟/计时独立登记，不直接混算加速。

仍计入M39原10小时累计预算，作业有界、profile与非插桩计时分开。此授权不包含改驱动、系统服务、GPU时钟或终止他人任务；baseline、投机生产、旧失败和四科学权限保持。

2026-10-21追加边界：用户正在协调约10分钟本机独占窗口，明确要求届时再用Nsight Compute。等待用户确认窗口后启动，不以检测到空闲自行启动本次profile；提前完成CPU准备，采集与清理控制在确认窗口内，原累计预算及旧失败保留。

2026-10-21随后用户更新：等待协调期间也持续查看本机，发现空闲即可启动本次NCU。此条替代上文“必须再等窗口确认”的临时约束；不终止他人进程，不在他人性能作业上叠加profile，仍按8分钟采集加清理有界执行。
