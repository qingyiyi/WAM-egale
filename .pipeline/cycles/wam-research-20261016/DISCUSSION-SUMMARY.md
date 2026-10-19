---
kind: discussion-summary
cycle: wam-research-20261016
updated: '2026-10-19T19:27:26.713126+08:00'
---
# 当前讨论与授权

用户确认：baseline不改，optimized从等价实现开始优化tile化Transformer；随后明确批准先完成隔离执行路径和tile子图。M31已完成静态子图及有限block验证。baseline及其实际依赖一并冻结，优化版独立DSO，不能间接改共享库污染对照。默认seq、tile与FFN1独立开关；旧局部融合不冒充新调度。

用户要求按Transformer self-attention/cross-attention/FFN真实DAG，借鉴MegaMoE思想，最终研究persistent warp/tile调度；不照搬MoE分解或SM100专有实现。当前只有显式attention/FFN1设备入口及同stream阶段子图，persistent/ready队列未实现。QKV/Wo/FFN2仍库调用，旧FFN2数值门不降低。

新full30因显存不足未测；旧M28 full30不自动继承。共享GPU只做正确性，性能仍等自然空闲。用户不需要重复协调窗口或无意义smoke。当前主线实现范围见PLAN M32，未自动启动后继或生产替换。

Long/Goal/Spatial首轮每臂500与Object既定队列继续；完整成功率已交付但不解除配对FAILED、曝光、组件问题。只main公开核对源码/HW/小结果，无权重/fixture/binary/凭据。

engineering_quality_screen=false；independent_test_complete=false；noninferiority_established=false；formal_performance_permission=false。旧FAILED/retention/NO_GO/paused72/96、MLP0、22缺收据/default A_off不改。

M17原10/21 16:56:31停新、17:11:31清理，18723000尝试/15GiB、每卡最多2worker；matched Target前200组同GPU1，不为填卡重复Target或破坏配对。BAAI仅~/cxz；AutoDL不再SSH、关机UNKNOWN。本机计时自然空闲、精度可共享；不杀他人/改服务/时钟。

投机源码/配置/运行依赖、原tile与state/rules保护，不建缺失cycle.yaml、不升级/旧sync覆盖HW。仅baseline/optimized两份主实现；baseline及依赖冻结，优化只在optimized。无自动新训练/120/独立TEST/新TF正式45chunk/第二环境/大融合。中文正文会话交飞书。10/22 go/no-go、10/23冻结、10/28主表、11/01 18:00中文包。


原授权与完整讨论见[归档](history/M31-before-summary-20261019/DISCUSSION-SUMMARY.md)。
