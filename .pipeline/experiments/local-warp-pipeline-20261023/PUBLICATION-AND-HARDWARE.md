# LingBot-VA 正式发表及测试卡建议（2026-10-23）

OBSERVED：原版 Causal World Modeling for Robot Control 已收入 RSS 2026（Sydney，July 13–17），DOI 10.15607/RSS.2026.XXII.016。
正式来源：https://www.roboticsproceedings.org/rss22/p016.html
PDF：https://www.roboticsproceedings.org/rss22/p016.pdf
官方代码仓库：https://github.com/robbyant/lingbot-va （本轮仅核验发表信息，不用 main 替换本地科研锚点）。

正式 PDF 第8页 Inference latency：单 RTX 5880 Ada，约0.5秒/闭环推理，5 video-denoising steps、10 action-denoising steps、16 action frames。此新证据更新此前未在旧版本定位到硬件的 UNKNOWN；不是我们的20/50配置，也不能将该绝对时间当成当前 shared-backbone checkpoint 的同条件 baseline。论文页面摘要出现 CauVA 命名，而正式PDF仍用 LingBot-VA；标题、作者和官方仓库对应。

RECOMMENDATION：当前主实验继续现有 A100-SXM4；先完成同卡、同输入、同缓存边界的强基线比较与三档KV验证。若要主张跨硬件适用，后续可增加 H100 一个平台（不是本轮执行/采购授权）。不必为了凑显卡数量分散工作；A100 PCIe/SXM不算跨架构。

Ada 当前不适合直接迁移：本轮 persistent kernel 实测 shared112384B=109.75KiB/CTA，超过 NVIDIA Ada 每block99KiB上限。需要改布局和重新核验。Hopper 每block227KiB，但需重编译、验证资源和数值，不自动继承A100结果。
官方硬件依据：
https://docs.nvidia.com/cuda/ada-tuning-guide/index.html
https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html

以上是主平台与可选可移植性安排，不是会议要求、录用保证或新增预算。

# 相关工作硬件补查

|工作|关系|明确的推理硬件|原文|
|---|---|---|---|
|Flash-WAM|直接蒸馏发布的LingBot-VA|单L40S，训练H100|https://arxiv.org/html/2606.05254v1 §4/Appendix A|
|WAMJET|直接优化LingBot-VA，作者项目页|H100；不要把其他模型的B200实验移给LingBot-VA|https://liulixinkerry.github.io/WAMJET/index.html|
|Fast-WAM|独立WAM，相关对照|RTX5090D V2 32GB|https://arxiv.org/html/2603.16666v1 §4.1|
|AHA-WAM|独立WAM，相关对照|RTX5090D|https://serene-sivy.github.io/aha-wam/static/aha-wam.pdf §4.1|

RTX5880Ada：Ada架构专业工作站卡，48GB GDDR6 ECC、285W、PCIe4.0x16。原厂https://www.nvidia.com/en-us/products/workstations/rtx-5880/ 。5880是型号，不是架构代号。各工作模型/去噪步骤/计时边界不同；这里仅核验硬件，不横排绝对延迟或加速倍数。

# 用户资源修正
用户当前没有H100；可讨论H800作为Hopper第二平台，不要求寻找H100。H800不是已接入/授权新作业。当前kernel shared109.75KiB超过SM120每CTA99KiB，5090D适配需压缩布局；Hopper上限227KiB。此理由针对当前整层kernel，不意味着5090D不能运行WAM或一般性能更差。若后续选择机器，建议有独占窗口及NCU计数器权限。
来源：https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html 与 https://docs.nvidia.com/ai-enterprise/release-8/latest/infra-software/vgpu/reference/hopper-h800.html 。
