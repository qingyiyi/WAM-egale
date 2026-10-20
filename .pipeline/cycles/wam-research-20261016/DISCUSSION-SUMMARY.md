# 当前讨论与授权

用户明确希望找到快速GEMM源码直接用于现有整层kernel，避免额外kernel启动。[决定](../../memory/global/requirements/decision-继续算子优化并先核对当前瓶颈.md)已保存。这延续整个**单层**persistent/tile/warp职责目标，不是改用host cuBLAS多kernel或30层融合。直接在optimized迭代，baseline连同依赖冻结，不另开带读代码版本/研究ID。

本次已接入固定CUTLASS SM80设备主循环并完成有限门/计时。可描述9签名有限逐位和一次compute launch，不能说已达Torch速度；B14仍慢2.54–3.02倍。当前仍255register/thread，warp1/2在大M路径空闲，原语内部流水不等于最终MegaMoE式独立角色流水。后续先定位新瓶颈再做ready填尾波，强cached baseline不能变弱。

原始action用原算子。B14是14个Q16逻辑实例，不是Q112序列；不共享private KV。baseline、旧投机源/配置、原tile/HW、state/cycle/rules不改。新ready未重验，默认False。当前总1497/1500block，无后续GPU作业；旧FAILED和预算起点保留。

RoboTwin用户授权修复后接续原有限迁移及首case视频入口补跑仍在原范围。六FIT身份Target已5成功1完整失败并清理，旧0teacher/0动作PARTIAL保留。TF采集/训练/DEV未实现齐，不声称自动排队。原A100/535配置、rt/32spp/path8/oidn保持；bank只有张量核验，完整任务等价未知。

BAAI所有写入限~/cxz；AutoDL已归还不连接；不改系统驱动/服务/时钟或杀他人。原GPU0≤8h、GPU1≤24h、准备10/20 18:20北京及各stage上限不重置。新GPU阶段需明确有界预算，不能从旧余额自动无界接续。

TF/normalizer/tau/selector/default A_off、旧FAILED/retention/paused72/MLP0/22缺收据和四项质量false保留。LIBERO既定6000局不是独立TEST；单层精度不转借为闭环质量/正式性能许可。PF2不是bonus或物理回滚。中文稿用户飞书整理，不新增论文初稿。

只main发布源码/HW/小证据；模型/PT/KV/图像/DSO/工具链/私有SASS/凭据不公开。10/22 go-no-go，10/23冻结，10/28主表，11/01 18:00老师证据包，不保证录用。[此前讨论](history/device-gemm-followup-start-20261020/DISCUSSION-SUMMARY.md)保留。
