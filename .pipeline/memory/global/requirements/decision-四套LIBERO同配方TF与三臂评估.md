---
kind: decision
name: 四套LIBERO同配方TF与三臂评估
level: constraint
scope: experiment:libero-four-suite-20261017
status: active
updated: '2026-10-17'
supersedes:
  - '三个小套件仅作为尚未获准的建议；本轮用户已明确授权'
sources:
  - '用户要求充分利用目前GPU，在Spatial/Object/Goal三个权重上用相同方法训练draft，测试Target、TF、TF+PF2的成功率、trace长度与E2E加速比'
---

# 新增三个小套件的独立TF训练及完整三臂对照

用户明确批准将Spatial、Object、Goal三个Wjjjh社区checkpoint加入研究。各套件沿用同一TF64结构、固定训练配方与预算，使用对应Target监督训练自己的draft；随后比较同套件Target、TF连续、同权重TF+PF2。不能把Long draft直接当作已适配，也不能用发布者或官方论文成功率替代当前社区Target实测。

现有Long继续，不改其冻结权重、原48小时账本或旧失败。新三套使用独立namespace、数据分区、normalizer、清理与预算；不增加架构搜索或训练大Target。采用每套10任务×50初态×三臂共1500局，三套4500局；训练采集额外episodes与成本单独登记，失败全部保留。原旧NO_GO/retention/default及四项false不改，不自动把含FIT/DEV的全量面板称为独立TEST。

充分使用当前已授权的本机和现有双卡AutoDL。共享资源质量可并行，独立模型/环境/KV/RNG与端口/输出分离，不超过实际显存/CPU/存储。AutoDL不做smoke、独立测试或SHA，包含隐式loader；沿源码直接完整任务并inline审计。所有臂同等使用完整512-token BF16优化reset。无新租机购买授权。

并行质量中的墙钟仅探索性成本。E2E正式加速仍按既有质量放行、无争用窗口、同硬件同边界测量；保存中间状态的replay只支持其对应verifier/chunk边界，不能替代完整机器人任务。具体生命周期、forward、存储与阶段输入清单见本实验PLAN，不重置旧账本。
