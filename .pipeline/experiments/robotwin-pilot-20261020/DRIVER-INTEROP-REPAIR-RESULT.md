# RoboTwin原始渲染恢复：A100可运行，客户端库选择已修复

2026-10-20，北京时间。状态`ORIGINAL_EXPERT_AND_OBSERVATION_RESTORED`。本次是在原A100-SXM4-40GB上做固定官方场景的完整专家运行与相机读取；0模型policy episode、0teacher forward，不是Target成功率验证。

## OBSERVED

实际检查发现，原CLIENT-ENVIRONMENT的LD_LIBRARY_PATH包含`/usr/local/cuda/compat/lib.real`和`/usr/local/cuda/compat/lib`，解析到CUDA13.2附带的**595.45.04 libcuda**；而系统内核、Vulkan/GLX驱动为**535.161.08**。

同一诊断脚本固定`place_object_basket / seed10000`，原rt shader、32spp、path depth8、OIDN2.3.3、物理、相机与GPU1绑定。仅对照子进程库路径；两臂同样启用OIDN_VERBOSE=2。完整调用顺序是官方setup→play_once至自然专家终态→close→同seed重新setup→get_obs→close。所有写入限~/cxz。

|attempt|实际加载libcuda|完整专家|重置后原相机读取|GPU生命周期秒|终态|
|---|---|---|---|---:|---|
|base-1|系统535.161.08|成功|四路RGB成功|42.208|COMPLETE，exit0|
|inherited-1|compat595.45.04|成功|OIDN非法访存|38.966|FAILED_OR_PARTIAL，guard SIGTERM|
|base-2|系统535.161.08|成功|四路RGB成功|38.172|COMPLETE，exit0|

两次base是同一场景的独立进程复核，不是两个独立任务样本。四路head/left/right/front RGB各240×320×3、uint8，非恒定图像，原始观察保存在远端；小型统计见各attempt的OBSERVATION-SUMMARY.json。此处未比较与RTX逐像素相等，也未证明全50任务稳定性。

运行时`/proc/<pid>/maps`同时保存了libcuda、Vulkan/GLX以及OIDN主/core/CUDA设备库，确认两种加载路径不同，OIDN版本一致。新版诊断把故障进一步定位到原相机观察阶段；旧粗粒度“首场景失败”记录保留。

## 解释与修复

[NVIDIA前向兼容文档](https://docs.nvidia.com/deploy/cuda-compatibility/forward-compatibility.html)明确列出CUDA–Vulkan互操作不受cuda-compat支持。现场受控对照支持：**本次故障与渲染客户端加载595兼容libcuda、搭配535图形驱动有关；改回配套的535用户态库即可恢复这个原始场景。** 没有进一步定位到某条底层驱动指令，不外推所有非法访存的原因。

已实际修改`/root/cxz/robotwin/runs/CLIENT-ENVIRONMENT.json`的LD_LIBRARY_PATH：去掉上述compat目录，前置`/root/cxz/robotwin/deps/system-driver535-links`。该目录仅含指向系统已经安装的535 libcuda/ptxjit/nvvm文件的用户目录符号链接。未安装、覆盖或重启系统驱动；两卡nvidia-smi仍报告535.161.08。未改shader/去噪/物理/相机，未改模型server环境、算子或原LIBERO环境。原客户端配置在远端base-2目录备份，修复收据为APPLIED-REPAIR.json。

**当前无需为这个故障更换4090。** 已有证据证明A100能运行该原始场景；[RoboTwin官方](https://robotwin-platform.github.io/doc/common-issue/index.html)仍提示A100/H100可能在部分场景挂起，[SAPIEN的RTX6000 Ada案例](https://github.com/haosulab/SAPIEN/issues/258)也存在相同OIDN错误，因此显卡型号本身不是兼容保证。将来如果真实任务账本显示渲染慢，再考虑RTX承担模拟器、A100承担模型推理；这是建议，没有租卡行动。

## 资源、保留与下一步

本次3attempt合计119.345351秒，低于1800秒修复子预算；计入原24h后累计614.666000秒，原10/20 18:20:28北京准备截止未重置。失败attempt的worker状态停在RUNNING，因为guard因致命日志终止；guard终态及SIGTERM原样保存，不伪造worker正常退出。

最终现场两卡均0MiB/0%，无compute进程，3个guard及其所记录worker身份均无残留。证据压缩传回23,311B，不传权重/图片/完整环境。旧OIDN两次失败、科学四项false、原预算与所有LIBERO结果保留。

下一步是按原登记先完成RoboTwin Target首6条完整任务，核对动作接口、真实延迟与长期渲染稳定性，再决定采集/训练接续。本次没有自动启动174局链，不把专家成功当作LingBot-VA成功。

证据：[本次计划](DRIVER-INTEROP-REPAIR-PLAN.md)、[base-1](driver-interop-evidence/driver-interop-base-1/GUARD.json)、[旧路径重现](driver-interop-evidence/driver-interop-inherited-1/GUARD.json)、[base-2](driver-interop-evidence/driver-interop-base-2/GUARD.json)、[实际修复](driver-interop-evidence/driver-interop-base-2/APPLIED-REPAIR.json)、[最终资源](driver-interop-evidence/driver-interop-final-resource.json)。执行源为render_driver_scene.py/render_driver_guard.py，实际每attempt源快照保存在对应SOURCE，第三次仅改输出namespace。
