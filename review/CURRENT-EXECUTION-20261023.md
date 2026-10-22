# 2026-10-23当前执行证据

外层FFN流水内联＋12个M32尾块K64已有限验证并采用。原Torch完整单层warm Graph/自有：**1.2426 / 1.7780 / 2.8859×**（B14，KV288/1152/2160），不是30层、chunk或E2E；双方同最终Cross K/RMS与V缓存。不是相对旧最强guarded路径的新增收益。

[完整阶段表](../.pipeline/experiments/single-layer-async-gemm-20261020/ffn-pipeline-inline-20261022/STAGE-TABLE.md)包含计算/公式、tile数、每tile工作量、warp职责、同步与Torch比值。当前108CTA×8warp，3408/3480/3408数学tile，27grid。阶段软件诊断两进程资源同、整层扰动≤0.462%；自有核内区间vs原Torch对应热阶段，边界不同，不作端到端分摊。FFN四段合计约1.01/1.01/1.03×，六小投影约0.68–0.74×。精确生产独立FFN仍UNKNOWN，不能择优引用编译上下文变化的重放结果。

两项改动各9真实签名/36结构、225完整调用有限逐位/状态通过，正式闭环四权限和production_admitted均false。旧FAILED、paused72、retention、review/RESULTS.json原身份不变。

[改动、负结果、输入及精度边界](../.pipeline/experiments/single-layer-async-gemm-20261020/ffn-pipeline-inline-20261022/RESULT.md)；[资源](../.pipeline/experiments/single-layer-async-gemm-20261020/ffn-pipeline-inline-20261022/RESOURCE-AND-STOP-REPORT.json)。累计15315.883/36000秒，算子owned已清理。用户确认远端无GPU硬件计数器权限，不再尝试提权/驱动改变；本机本次NCU0启动。

RoboTwin固定1800局在GPU1继续，GPU0保留算子。10/23 00:45完成15/1800，T/C/P各5/5成功，5配对审计通过；属于早期扩大DEV状态，不是总体SR或独立TEST。600有效初态和优化reset bank已备齐，队列自动接续，当前open_microwave-valid000-P。仍受160GPU小时/10天约束，旧24小时与失败不改。

下一步先审查六个小投影的函数边界、队列/两处grid与写回开销；现有数据不能断言全是同步。不要恢复大范围ready队列或为了得到PASS更改质量规则。

原始大tensor、checkpoint、DSO、SASS二进制/大文本和SSH配置不在公开镜像；源码、脚本、小型原始结果及摘要已同步。公开仓库是进度中转，不是完整可执行环境。
