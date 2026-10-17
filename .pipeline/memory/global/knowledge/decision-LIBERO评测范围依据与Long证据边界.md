---
kind: decision
name: LIBERO评测范围依据与Long证据边界
level: reference
scope: cycle:wam-research-20261016
status: active
updated: '2026-10-17T17:22:00+08:00'
supersedes: []
sources:
  - '用户询问是否只做Long10、不做90以减少工作；这仍是调研问题，不是强制取消旧授权'
  - 'LIBERO官方README：90用于预训练、10用于下游终身学习评测'
  - 'LingBot-VA原文2601.21998v1 §4.3.2/Table2'
  - 'CoT-VLA CVPR2025官方论文页面和作者项目页实验表说明'
  - 'FLASH原文2605.13778v1 §4.1/4.2（预印本）'
  - 'Seer作者官方代码和实验导航'
  - '2026-10-17官方发布列表及Wjjjh三个固定revision模型卡/API复核，见三套件可用性RESULT'
---

# LIBERO90并非这些论文的通用必测项

此记录固定本轮已核实的证据口径（level reference），不将研究建议变成用户已决定的实验范围。

LIBERO官方原始设计把100拆为90预训练和10下游评测。90可被用于其他研究，但不是这些VLA论文的通用必测项。现代常见四套各10任务也不等于LIBERO100。

已核原文：LingBot-VA主表和FLASH使用Spatial/Object/Goal/Long（或LIBERO-10）四套各10task；CoT-VLA CVPR2025作者项目页同为四套，每套500初态、三轮统计。它们不是LIBERO90，也不是只抽10条episode。LingBot-VA原文每套每seed500局、三seed，当前Long每臂500局只相当于一个固定evaluation seed的完整套件覆盖。

Seer的LIBERO分支只列Long，另有CALVIN与真机；这是“LIBERO部分可以聚焦Long”的先例，不能据此推出“唯一实验只有Long也必然足够发表”。我们不能从论文没有跑90就断言CVPR有某个统一最小验收规则。

研究建议：90不列当前必做依赖，优先收口Long的冻结Target/TF/PF2质量、800失败罚分、轨迹分布、有效提交成本与分层/完整任务性能。若声称跨任务套件、跨WAM普适性，仅同一Long checkpoint证据较窄；质量与系统成立后再考虑一个有可信Target的小套件。三套社区权重不是90，不自动改用户范围或启动新训练。

按50初态×三臂计，Long1500局，90另13500局；不做90省去90%的Long+90计划episode数量，不保证同比墙钟节省。还避免新增Target适配、90 prompt bank、teacher数据与draft训练。原Long质量未过/四false/FAILED保持；增加或减少套件都不能替代质量门。

原文：[LingBot-VA](https://arxiv.org/html/2601.21998v1)、[CoT-VLA官方CVPR](https://openaccess.thecvf.com/content/CVPR2025/html/Zhao_CoT-VLA_Visual_Chain-of-Thought_Reasoning_for_Vision-Language-Action_Models_CVPR_2025_paper.html)、[CoT-VLA作者评测说明](https://cot-vla.github.io/)、[FLASH](https://arxiv.org/html/2605.13778v1)、[Seer官方](https://github.com/InternRobotics/Seer)。本地核查材料：投机采样/results/runtime/p0-libero-eval-scope-literature-cpu1-v1-20261017-r1。

## 三个小套件的公开资产

官方当前下载表及HF作者列表未列Spatial/Object/Goal权重；社区Wjjjh三套固定revision均有公开transformer、config及独立norm_stats，模型卡明确非官方，从base训练5000更新。发布者自报Object99.07%、Goal96.73%、Spatial95.33%，上述15:53快照时本项目尚未下载或复现，不能继承这些成功率；随后Object/Goal已完整下载，实测仍待执行。TF64结构可复用，Long draft跨新teacher/归一化的质量仍未知。每新套件三臂各500局=1500局，三套另4500局。

截至15:53的询问本身未批准三套。随后用户已明确批准三个checkpoint各自同方法训练和三臂完整评估，现行范围见[新约束决定](../requirements/decision-四套LIBERO同配方TF与三臂评估.md)；先只补Object的建议未被采用，不取消90的原授权。固定发布身份、模型卡与7次HTTP读取证据见[三套件资产核查](../../../../投机采样/results/runtime/p0-libero-three-suite-availability-cpu1-v1-20261017-r1/RESULT.md)。
