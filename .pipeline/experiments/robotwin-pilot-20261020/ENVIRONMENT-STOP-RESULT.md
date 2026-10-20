# RoboTwin 环境阶段结果：原渲染链受阻

2026-10-20，北京时间。`STOPPED_INFRASTRUCTURE_NOT_MODEL_FAILURE`。原 Target 复现、84 局采集、训练和 90 局 DEV 均未开始。不能把环境准备或专家尝试当成模型成功率。

## OBSERVED

固定 LingBot-VA `7c6ffa9bfc4b83582cafc860fab4c82cc7deeeeb`、RoboTwin `2eeec322d95799f537cbfe5f291a8220d965ccb8`、官方 RoboTwin Target HF `8c9dea8abbc5c91cc9e18bc3264b8915083bbe70`。模型 transformer 三分片已取得；公共 text/tokenizer/VAE 通过固定发布方文件元数据核对后复用，没有独立运行大文件 SHA。资产和私有客户端环境、CuRobo 0.7.8/SM80 构建完成，所有写入限 `/root/cxz/robotwin`。

GPU1 A100-SXM4-40GB/driver535.161.08 的 SAPIEN capability 查询报告可渲染、支持光追，且按 PCI `0000:4e:00.0` 绑定。**真实创建首个 `place_object_basket` seed10000 场景时，OIDN 报 `an illegal memory access was encountered` 并挂起**，尚无完整专家终态或 policy 请求。

两次真实环境尝试分别使用原随 SAPIEN 的 OIDN2.0.1，以及依据官方问题记录在私有环境安装的 OIDN2.3.3；均保留 rt shader、32 spp、path depth8、oidn、相机与物理设置，均失败。后者的 import-only 动态库映射确认加载2.3.3主/core库；该元数据查询不是场景通过证据，也未保存失败进程的完整运行时库映射。

|尝试|终态|GPU生命周期秒|完整policy|teacher forward|
|---|---|---:|---:|---:|
|prefilter-control-4 / prefilter-1|OIDN非法访存，显式停止|450.318|0|0|
|prefilter-control-5 / prefilter-2|OIDN非法访存，guard停止|45.003|0|0|
|合计|环境阻塞|495.321|0|0|

两个 worker 以 PID+birth/pgid 身份核对终止并被 guard 回收。第二次的 guard 同时留下 `timeout=true`，实际先检测到 fatal 日志而停止；原字段保留，不能把它改写为自然超时或 exit0。下载直连超时、CuRobo 头文件等前期实施失败另有原日志，未进入 GPU episode。

截至[资源快照](remote-evidence/STOPPED-RESOURCE-SNAPSHOT.json)，未观察到本命名空间活动进程或 GPU1 compute context。环境/资产/模型/源码/依赖/缓存合计远低于100GiB帽；runs也低于12GiB帽。实际字节见快照，未删除失败或占用GPU继续等待。

## HYPOTHESIS / UNKNOWN

[RoboTwin官方常见问题](https://robotwin-platform.github.io/doc/common-issue/index.html)列有 Ampere/Hopper 挂起；[SAPIEN维护者的相近驱动案例](https://github.com/haosulab/SAPIEN/issues/219)说明部分驱动报告支持光追但启用后仍可挂起。[OIDN兼容库修复参考](https://github.com/haosulab/SAPIEN/issues/243)在本机并未解决问题。

这些资料支持环境/驱动互操作是待查方向，**没有独立证明本次根因就是驱动**，更不能据此断言A100全都不支持RoboTwin。尚不知道关闭光追/去噪后的图像变化、与官方 Target 评估的等价性或成功率；本轮未运行这些替代条件，也未修改系统驱动。

## RECOMMENDATION / 依赖

原协议迁移链停在原 Target 之前，暂不做新采集/训练/DEV。恢复需一台已能运行固定 RoboTwin 原 rt/oidn 渲染的环境，或单独确认一个明确版本的兼容修复；不以更换任务/seed、静默关闭去噪、删相机或缩短 episode 绕过。

新机器/驱动变更不在这次私有环境授权中，未采购或修改系统。现有模型、资产、源码和修复日志已保留，迁到兼容环境可复用；单局耗时及174局完工时间仍 `UNKNOWN`，不能从LIBERO外推。准备截止仍10/20 18:20:28北京、已花495.321秒计入原24h生命周期，不重置账本。

四项权限继续false。原LIBERO代码、配置、质量结果和算子baseline未改。
