# 资源与停止

用户新授权预计算及逐阶段计时；原累计十小时不变。窗口2026-10-22T16:25:20.330930+08:00—2026-10-22T17:10:20.330930+08:00，修复profile沿同一绝对截止，没有续期或重置起点。

GPU worker生命周期合计127.569秒；从登记至最终清理保守记650.662秒（包括传输、CPU修复等待与清理），累计8349.913/36000秒。CUDA active未知。原始guard的FAILED_OR_PARTIAL与首profile FAILED全部保留；修复补充位于profile-r2，最终组装报告完成，未覆盖旧终态。

两项CPU准备失败：正常版pybind中Tensor名字解析冲突；诊断构建缺少代码生成模块搜索路径。均在CPU上修复。随后正常/诊断编译各约300秒。首GPU profile漏记录入口/出口两个同步，断言中止；[原因](profile/diagnostic-attempt1/ROOT-CAUSE.md)；修复仅补插桩，重新编译诊断版，正常精度/计时未重跑。1项GPU诊断失败、无timeout。

2026-10-22T16:36:10.992744+08:00全部owned PID+出生身份退出，未杀他人进程；[首阶段清理](FINAL-CLEANUP.json)、[最终清理](profile-r2/FINAL-CLEANUP.json)。旧M40截止/预算及四科学权限false不变。本轮没有新机器人或30层测试。
