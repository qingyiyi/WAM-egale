# RoboTwin执行进度

- [x] 新授权/有限预算记录
- [x] 固定源码、官方 Target/公共组件来源、资产、私有环境及 CuRobo 构建
- [ ] 原始渲染链受阻：OIDN 2.0.1/2.3.3 均首次场景非法访存；两次owned worker已停止，共495.321秒、0policy/0teacher；[环境止损结果](ENVIRONMENT-STOP-RESULT.md)
- [ ] 原Target首6局及真实估时
- [ ] draft/verifier/reset适配
- [ ] 84局采集与固定训练链
- [ ] 90局完整三臂DEV与汇报

四项质量权限保持false。

固定的六任务、19 场景/task 清单尚未产生，不声称首六局已运行。GPU1 设备 capability 查询支持渲染并已绑定正确 PCI，但真实场景 OIDN 仍失败；二者不能互相覆盖。仅修改用户目录内兼容依赖，保持 rt shader、32 spp、path depth8 和 oidn 选择，最终依赖版本必须在新场景清单前冻结。

CPU 准备截止仍为 2026-10-20 18:20:28 北京，GPU 总生命周期 24h，不重置。状态/失败/清理原件：远端 `/root/cxz/robotwin/runs/{configure-3,prefilter-control-4,prefilter-1,prefilter-control-5,prefilter-2,GPU-LEDGER.json}`。所有写入限 ~/cxz，无系统或驱动变更；AutoDL 未连接。
