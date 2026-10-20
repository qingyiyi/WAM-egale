# 原场景渲染链修复：进程内库选择

用户2026-10-20已恢复环境排查，明确不修改系统驱动。沿当前M40记录，保留原两次失败和原预算；不是新采集、训练或模型测试。

## 可检验假设

现有CLIENT-ENVIRONMENT继承`/usr/local/cuda/compat/lib.real`，该目录实际指向CUDA13.2提供的595.45.04用户态libcuda；系统Vulkan/内核驱动为535.161.08。NVIDIA官方前向兼容文档明确CUDA–Vulkan互操作不受支持。旧进程完整库映射缺失，因此当前仅为待验证假设。

只改变RoboTwin子进程的库搜索：移除compat目录，通过~/cxz内链接指向已安装的535 libcuda/ptxjit/nvvm库；不覆盖系统文件。固定SAPIEN、OIDN2.3.3、GPU1、rt/32spp/path8/oidn、物理/相机、place_object_basket seed10000。

## 作业与边界

- 最多3个有明确差异的attempt；合计GPU生命周期≤1800秒，单attempt≤600秒并预留清理；仍计原24h帽、原10/20 18:20:28北京准备截止不重置。
- CPU worker2核，guard1核；新增持久化≤256MiB。不做模型forward，不启动原174局链。
- 先在固定原场景完整执行官方专家，再按原参数重新setup并读取完整相机观察；每attempt至多1个完整专家case。不挑seed，不截短机器人episode冒充质量结果；基础设施中断记PARTIAL。
- inherited/base两种库选择共用同一脚本；基线仅在归因必要时复现，不无条件重复已知失败。
- 保存加载库映射、准确CUDA设备与驱动、阶段、原日志、专家自然终态和相机观察；OIDN库导入成功不等于真实渲染通过。
- 每个attempt登记输入/代码路径、资源、终态、失败原因、PID+birth身份、退出方式与清理。只回收本attempt身份匹配进程。
- 如果host库选择解决非法访存但出现后续依赖问题，分别报告，不把渲染修复夸大为完整评测可用。

源证据：[NVIDIA前向兼容](https://docs.nvidia.com/deploy/cuda-compatibility/forward-compatibility.html)、[RoboTwin官方常见问题](https://robotwin-platform.github.io/doc/common-issue/index.html)、[RTX Ada同报错](https://github.com/haosulab/SAPIEN/issues/258)。验收后给出A100继续使用或RTX环境迁移建议，不预设购卡必要。
