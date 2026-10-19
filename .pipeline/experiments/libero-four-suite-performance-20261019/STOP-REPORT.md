# 资源与止损

更新时间2026-10-20T02:38:05.127898+08:00；终态/进度：RUNNING_PARTIAL_TIMING_PANEL。

M35完成301/360完整episode；Object首轮在CPU白名单前置检查失败，0episode/0teacher，原失败保留且不重试。仅原未启动2/3已运行，原面板最大330，不把缺30计为算法失败或缩小原预期。

已保存episode COUNTER的target尝试合计205233；这不是把B2与B14当等成本。已结束runner phase生命周期合计4.8027 GPU小时口径（各卡各phase分别相加），缺少独立失败启动/排队成本，不能当全预算账。

更保守的原父窗口：两卡均从原T0计到本快照，合计6.0507 GPU小时，包含已释放卡的空闲、等待、准备、失败和清理空档；低于12 GPU小时帽。未用上限约5.9493小时，不自动转授新作业。它不是CUDA active，也不与M34重叠账本再次相加为租卡费用。

原硬限仍为10/20北京05:34:31停新、05:44:31清理；不延长。CUDA active总秒UNKNOWN，未查租卡实例计费或执行实例关机。每phase独立guard/出生时间/真实exit收据保留；最终GPU空闲与controller身份另看最后资源核对。

M34已关闭：600完整block调用（含参考/校验/计时）、0full30/episode；原数值/速度NO-GO，全部算子owned退出。详细生命周期、编译/依赖/CPU核对失败见[算子止损](../single-layer-warp-specialized-20261019/RESOURCE-AND-STOP-REPORT.md)。

停止方向：本轮不再布局/register/grid扫参，不接新kernel到生产，不增加环境、训练、独立TEST或旧45chunk，不把PF2默认打开。新任务/样机需按后继问题重新登记，不能重置旧预算。旧FAILED/四false/retention/default A_off保留。
