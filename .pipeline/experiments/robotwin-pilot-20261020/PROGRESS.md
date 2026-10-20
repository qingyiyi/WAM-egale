# RoboTwin执行进度

2026-10-20用户先暂缓，随后明确恢复环境排查且保持驱动。现已[修复原场景渲染](DRIVER-INTEROP-REPAIR-RESULT.md)：535配套库两次完整专家/四路相机通过，595 compat对照重现失败；客户端路径已修复，驱动不变，旧记录保留。本次GPU诊断均结束，原Target与采集链未自动启动。

- [x] 新授权/有限预算记录
- [x] 固定源码、官方 Target/公共组件来源、资产、私有环境及 CuRobo 构建
- [x] 原始场景渲染恢复；旧OIDN两次失败495.321秒保留，新增对照119.345秒，原总累计614.666秒；0policy/0teacher。
- [ ] 原Target首6局及真实估时
- [ ] draft/verifier/reset适配
- [ ] 84局采集与固定训练链
- [ ] 90局完整三臂DEV与汇报

四项质量权限保持false。

固定的六任务、19 场景/task 清单尚未产生，不声称首六局已运行。GPU1 设备 capability 查询支持渲染并已绑定正确 PCI，但真实场景 OIDN 仍失败；二者不能互相覆盖。仅修改用户目录内兼容依赖，保持 rt shader、32 spp、path depth8 和 oidn 选择，最终依赖版本必须在新场景清单前冻结。

CPU 准备截止仍为 2026-10-20 18:20:28 北京，GPU 总生命周期 24h，不重置。状态/失败/清理原件：远端 `/root/cxz/robotwin/runs/{configure-3,prefilter-control-4,prefilter-1,prefilter-control-5,prefilter-2,GPU-LEDGER.json}`。所有写入限 ~/cxz，无系统或驱动变更；AutoDL 未连接。
