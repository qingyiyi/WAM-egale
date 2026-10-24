# M41资源、失败与清理报告

终态：COMPLETE_WITH_LIMITATIONS_AND_RECORDED_FAILURES。所有自然任务与数据分析结束；不是质量/正式性能PASS。

## 实际消耗

- 原T0 2026-10-25T01:15:33+08:00，本轮目录删除北京时间2026-10-25T06:01:46.687919+08:00；两者间17173.688秒（4.770小时），这是GPU作业及远端资产收口区间。启动、编译/依赖、等待、失败、传输及清理均在这一原生命周期内，不重置。原10小时/10:55工作停止/11:15清理边界保留。
- 5090单模型worker，CPU模型0–1、client2–3，准备/传输在4–7范围，未新增租机。GPU active时间UNKNOWN，不能把CUDA事件跨度或host时间称实际GPU active。
- 共141次client进程启动尝试：135条完整自然终态，2条已teacher执行但0环境动作的Long PARTIAL，4条teacher/动作均0的client准备失败。另有未启动client的模型准备失败。所有准备失败仍计上述生命周期，不增加独立样本。
- 原natural cap138；完整终态135，连同2条已执行PARTIAL为137。141次client进程启动不能改称141条独立自然episode，也不能隐去准备失败。主表只用44组三臂132局；当前引擎2条未配齐Long完整局和旧FSDP的1条完整Target保留在外。
- 全部server计数含旧/新与失败：teacher尝试95597、返回95596，低于原600000尝试上限。B2与B14保留原始类型，不视为同成本。完整episode timeout记录0条。
- 最终回传1664文件、88104231B（84.023MiB），低于3GiB。原始数据/代码在raw/，无模型权重。

## 失败与限制

- Long原FSDP task0/T完整观测保留，不与后来强noFSDP主比较混用；C task0旧绝对source路径失败21teacher，原panel计数滞后，单独收尾取回真实累计1277/差21，见raw/long/ENGINE-V1-FINAL-COUNTERS.json，不修改原panel。
- C task1观察器丢失公开forward签名，正确触发lease撤销，23尝试/22返回、0动作；修复仅保存签名元信息。两条已执行PARTIAL不重跑，缺P0未开始，主比较Long只有8组三臂。
- 纯准备失败含EGL/用户态库/依赖、Robo动态AST行号与client父目录，以及Object bank绝对链接失效。Object旧attempt01为0teacher/0episode，原bank复制实际内容后新02首次启动完整任务；未重算T5或改数值。
- Object源bank为绝对符号链接，普通tar保留它；源导出器未来改dereference=True，原失败、readlink和native复制修复收据保留。
- 物理存储上限有一次明确违反：源Goal传输时文件系统在EOF之外预分配，owned du采样36G超过34GiB；真实连续peak UNKNOWN。身份核验后SIGTERM仅停自有rsync（exit20），writer关闭回收多余分配。原cap不放宽、不改旧失败；增加exact-size preallocate与物理du guard后续传。见STORAGE-OVERSHOOT.json、raw/GOAL-INPUT-PREALLOCATION-STOP.json及原transfer日志。
- Robo前分叉隔离FAILED，四科学权限保持false。有限Q32 attention16次exact不证明完整模型或B2/B14 teacher等价；reference计算/同步/落盘计入首个共享路径任务，未事后扣除。受控资产预传存在CPU/IO重叠，不作为正式无干扰论文计时。

## 清理与下一次复用

174个PID+出生身份核查；最终补发1个自有进程清理信号，活跃owned=[]，GPU compute contexts为空。各suite的server停止另有原收据。post检查删除根命令关联进程为空，原operator工作区/环境/activation实际存在。

最终native rsync传输完整性、不可变文件尺寸清单及每文件本地fsync核验后，删除整个/root/autodl-tmp/cxz/spec-speed-20261025，删除表观14504984691B（13.509GiB）。包含本轮全部新Target、draft/bank、venv、simulator、cache及临时文件。当前数据盘可用41.944GiB；原算子工作区保留，实例未关闭，不能把job退出当租卡关闭。

A100原模型、正在运行的质量队列、可复用源导出包及本机冷归档原件保留，不因本轮结束清理。下次需要时重新向5090迁入。专用本地SSH/agent的退出以最终私有transport收据为准，不进入公开镜像或泄露凭据。

四项：engineering_quality_screen=false；independent_test_complete=false；noninferiority_established=false；formal_performance_permission=false。原FAILED/retention/protected state/cycle/rules及原tile未修改。

固定证据：RESOURCE-AUDIT.json、REGISTRATION.json、FINAL-RETURN-ACK.json、FINAL-OWNED-PROCESS-CLEANUP.json、FINAL-REMOTE-DIRECTORY-DELETION.json、POST-CLEANUP-VERIFICATION.json。

截至CPU报告/HW与发布整理检查2026-10-25T06:29:00.133606+08:00，自原T0总墙钟18807.141秒（5.224小时），仍在原10小时内；不能把4.770小时GPU作业区间称为整个租卡账单或全部CPU整理时间。
