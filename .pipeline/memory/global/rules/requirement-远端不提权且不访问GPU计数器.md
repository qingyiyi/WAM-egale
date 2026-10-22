---
kind: requirement
level: constraint
scope: project:WAM-egale;host:BAAI-dx-calc1
status: active
updated: '2026-10-23'
sources:
  - 用户明确确认远端不能给予root管理权限，也不能访问GPU计数器
supersedes:
  - 远端NCU权限待平台管理员开放的后续建议
---
# 远端不提权且不访问GPU计数器

用户已向平台确认：BAAI远端不提供所需宿主机管理权限或额外容器capability，也不允许访问GPU硬件性能计数器。后续不再申请/尝试提权，不修改RmProfilingAdminOnly相关驱动配置，不更换或重启驱动，不重复运行被权限阻断的NCU采集。

远端仍可在现有授权内执行精度/状态验证、CUDA Event计时和软件时间线插桩；这些不能冒充GPU硬件计数器。需要NCU时，只在本机自然空闲且原资源约束允许时有界采集，远端速度与本机计数器分别注明机器和边界。无本机窗口时将新增硬件计数标UNKNOWN，不因此停掉可独立推进的研究。

SSH容器内显示uid=0不代表获准宿主机管理权限；本条记录平台能力边界，不修改已有作业预算、GPU0算子/GPU1机器人分工或科学权限。
