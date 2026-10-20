# 修复后接续：初态清单与原Target首六局

当前接续：control-7/prefilter-4已完成114个固定初态；control-6/first6-1旧失败保留，[源码依据](EXPERT-SELECTION-CONTINUATION.md)。first6-2首case在动作前因可选视频导出入口失败，0teacher/0动作。用户明确允许修正后补跑，first6-3沿原17:46:42北京绝对deadline运行，见[补跑记录](FIRST-CASE-RETRY.md)。

2026-10-20。用户明确恢复原六任务有界迁移；原T0、24GPU生命周期小时、18:20北京准备截止与旧FAILED全部保留，不增加174局总数。本次不启动新算子GPU实验。

## 当前可执行范围

1. `prefilter_guard6.py` → 原专家筛选，每任务固定19有效seed，按原计划8 Target/3 parent/3 current/5 DEV。GPU1、client2核/control1核，最多3600秒＋300秒清理、1200次专家尝试、2GiB；0 policy/teacher。使用已修复的535客户端库路径，原rt/32spp/path8/OIDN不变。旧两次失败目录不复用。
2. `target_first6_guard.py` 等上述清单完整且guard清理后，自动接续 `target_server.py` / `target_client.py`。固定六任务各valid00，均已分配Target FIT；不按模型结果选择或替换。
3. 原Target首六局结束/失败后停止并审查。**后续投机collector/verifier/train/DEV尚未完整实现，不声称已自动排队。**

首六局自身总墙钟上限7200秒（含等待、setup和清理），开每局前预留完整1800秒与900秒清理；每局teacher尝试≤8000，6局≤48000，持久化≤2GiB。等待无GPU上下文时另记CPU等待，实际GPU占用追加原账本，不重置。预算不足不启动下一局；任何基础设施/身份失败不自动重试。自然算法失败保留，仍允许固定下一任务。

## 原实现、RNG与数据

固定LingBot源码`7c6ffa9bfc4b83582cafc860fab4c82cc7deeeeb`、RoboTwin`2eeec322d95799f537cbfe5f291a8220d965ccb8`、Target revision `8c9dea8abbc5c91cc9e18bc3264b8915083bbe70`。独立wrapper，不覆盖原源码。

沿原完整client动作/缓存循环，替换专家选择及指令生成为冻结清单；现场比较完整初态/物理与观察，恢复该初态保存的client CPU/CUDA/Python/NumPy RNG。模型evaluation seed前瞻固定为0，每episode reset前复位；它不是训练seed20260923。保留官方video25/action50、末尾video/action缓存调用、原EEF/四元数/夹爪映射和任务step_lim。完整机器人任务不截断冒充成功率。

监督只观察实际Target action query，保存原CFG2方向和**真正传入原scheduler的方向**、原latent/previous direction/sigma/next-sigma/timestep、更新后状态和decoded action。钩子核对50次合法更新及teacher CFG标签身份，返回原运算结果。每局环境命令逐条保存；请求、teacher attempted/returned、终端调用分别计数。

这些小监督行没有完整历史KV/视觉条件，不声称可以在任意新latent上重新询问teacher；现阶段不支持递归同query重标注。它们不冒充反事实节点或独立TEST。

## reset优化

从同一原Target、原GPU BF16编码器对固定清单指令及空负提示生成完整512×4096 bank，存取逐位核对，之后T5移CPU释放显存；VAE与模型仍原GPU路径。所有在线请求只读该bank，miss报错不悄然换编码路径。bank构建无teacher forward，冷成本另列。当前只证明缓存张量拷贝身份；完整任务cached/uncached等价尚未实测，不能借LIBERO旧证书宣布RoboTwin已完成等价门。

关闭原调试用逐chunk相机/latent重复落盘和比较视频导出；保留有界初态、真实teacher监督、环境命令及终态。计时包含观察钩子与采集开销，只是资源/探索账本，不是正式E2E加速比。

## 清理与终态

所有新写入在`/root/cxz/robotwin`；GPU1 UUID保持原授权。guard记录PID、birth ticks及真实退出方式，只清理匹配owned进程，所有attempt保留。原24h账本追加；没有新租卡/驱动/系统服务/全环境/架构搜索或正式性能授权。四项科学权限保持false。
