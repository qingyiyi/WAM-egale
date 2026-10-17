# 给网页版 GPT：TF draft 的文献、官方实现与有限收口方案

请用中文审查原文与代码，为服务器 Codex 给出 TF draft 的有限收口方案。需要证据和可执行主线，不要猜根因或无界搜索。本文不是论文初稿或执行授权。

## 1. 首先绑定新版快照

仓库：<https://github.com/qingyiyi/WAM-egale.git>

- 新交接分支：`review/tf-draft-handoff-20261017`。
- 本轮 `snapshot_id` 必须为 `wam-tf-draft-handoff-20261017-paused72`。
- **只读用户提供的新固定 commit SHA**；未另贴则确认本文件所在提交的 SHA。分支仅用于定位，禁止跟随 main 或分支后续变化。
- 旧科学基线仍是 `b0dfac3a02f65de33a1217246b83003615166640`，只作历史比较；不能把该旧快照冒充当前结果。
- 开头列实际 SHA、snapshot_id 和缺失文件。无法访问 GitHub 就明说并索取缺件，不能假装读过；继续独立原文核查。

先读固定提交的 `review/START_HERE.md`、`review/RESULTS.json`、`review/WEB_REVIEW_PROMPT.md`（若存在）、`.pipeline/PROGRESS.compact.md`。按索引定位新版证据，缺件标 UNKNOWN，不用上游 main 替代本地契约。

重点读以下已同步源码，并为关键判断给出固定提交链接及行号：

1. `投机采样/experiments/draft_architecture_v1/models.py`：真实 TF/MLP forward、维度、残差头。
2. `投机采样/experiments/tf_effect_train_v1/{train.py,adapter.py,gpu_actual_cal.py}`：本轮有限训练、数据配方、loss、normalization、部署与 CAL。
3. `投机采样/experiments/tf_effect_collector_v1/collector.py`：teacher 从哪里采、哪些是实际访问状态、哪些只是反事实树节点。
4. `投机采样/experiments/tf_effect_campaign_v1/{protocol.py,server_entry.py}`，及同步的 continuation/report：完整任务、真实 dtype、接受路径和结果来源。
5. `投机采样/scripts/p0_batched_verify.py`、`投机采样/experiments/prefix_acceptance_v1/server_entry.py`、`投机采样/wam-p0-worktree/wan_va/modules/model.py`：B2/B14 teacher、sigma/timestep、CFG、KV、scheduler 与接受分数。
6. 索引中的同 query 数值诊断、训练/CAL、双 namespace 汇总、recorded-risk 组件诊断及暂停收据。缺原 tensor 不得声称独立重算。

## 2. 必须保持的当前事实与边界

全实验暂停，直到用户明确恢复；建议不是授权，不得自动续租或启动。

固定 DEV96 完成 **72/96 个完整机器人终态**：原 namespace70、新2；T28、TF_CONT20、TF_ON24、MLP_ON0/20。汇总 PARTIAL：72条机器人完整终态、50条逐局必要组件审计齐全，总体严格证书未建立。原 TF_ON22条缺停止时未写出的 `OWNED-CLEANUP.json`；全局 root/clock/owned 退出已独立证明，不能补造局部收据或改 strict gate。

新 namespace 在完整 task5 后按用户要求停止，原汇总的 source_terminal issue 保留，不能改称预算完成或共同 PASS。

4个已完成 known-risk TF_ON 全失败：task1/init29、task3/init26、task4/init26 为 T成功/TF失败；task5/init28双方失败。前三条 TF实际步796，失败罚分800，分别报告。fresh成功数/长度读新版 RESULTS，不推断未跑 MLP 优劣。

本轮事前允许的**一次新数据与等预算训练机会已使用**，包括 TF-cont、TF-on、MLP-on。新同 CAL 上 TF_ON 的 mean 局部误差较 TF_CONT 低约4.46%，但这不是闭环改善保证。现有同 query 诊断发现差异先出现在 teacher 路径，不能一开始归因 draft；时间编码单项对齐尚未恢复完整 velocity，具体比较边界必须读原报告确认。

TF64 为主候选，MLP只作匹配消融，不能牺牲质量。结构：53070参数；16 token×17维，state7、previous direction7（根为已有target方向，递归节点为父draft方向）、sigma、next sigma、t/1000；1层 pre-norm，D64/H4/FFN256、GELU、dropout0、学习位置编码、两组7维残差头。draft无图像/prompt cross-attention；冻结 WAM 仍有视觉/语言条件。

action latent 为7×4×4=112维，与 Transformer 隐藏维（例如代码中的3072）不同。7 candidate×CFG2形成外部 B14，各逻辑实例的 attention Q仍为16，不能描述成一个可跨候选任意注意的 Q112 序列；private KV 不默认共享。

接受分数 `abs(Δsigma) × ||v_draft − v_target||₂`，固定阈值 `0.007839680526450805`，BF16方向/FP32打分，仅旧局部校准。禁止改阈值、旧 FAILED/retention 放行。PF2是两个 speculative chunk 后一个 target chunk，未启用，与 bonus 不同，不恢复已偏离状态。

四项原权限始终 false：`engineering_quality_screen`、`independent_test_complete`、`noninferiority_established`、`formal_performance_permission`。TEST、120局、正式性能和本轮新系统实验均未启动。DEV非显著不等于非劣，不能用成功子集隐藏失败。少 forward/更早失败不等于 E2E加速；主要成本 `ΣT_round/Σg` 未采齐则 UNKNOWN。

## 3. 文献检索：先原文和官方代码

优先正式 CCF A/相关顶会，深核最相关3–5篇。线索：FlowCast（核 ICLR2026）、Spec-VLA（核 EMNLP2025）、Consistency Policy（核 RSS2024）、TS-DP（核是否仍为预印本）。逐一核题名/作者/正式状态/年份，预印本不当录用，机器人顶会不自动标 CCF A；未找到标 UNKNOWN。

检索连续 diffusion/flow policy 的 draft、蒸馏、consistency、schedule-aware/recursive 训练及验证。离散 token 仅作边界对照。证据限原文、正式论文页、作者/组织官方代码；二手解读只用于找原件。

每篇用一行摘要加一张证据卡，务必包括：

- 正式状态、原文、官方仓库固定 commit/tag、许可、checkpoint与依赖；有代码不等于有权重。
- draft/学生准确结构：输入输出、token化、层数、宽度、头数、参数量、时间编码、残差方式；共享 target encoder、复用 hidden/视觉特征或 cross-attention 的具体位置和推理成本。未公开项写 UNKNOWN。
- 训练集究竟来自公开离线示范、冻结 target rollout、模拟器 on-policy、teacher 反事实 query，还是混合；FIT/CAL/DEV/TEST 如何隔离；是否需要在 LIBERO 内采样，以及实验环境与训练数据是否同分布。
- teacher label 是 noise、velocity、score、x0、动作或已合成结果；CFG 合成在前还是后，sigma/timestep和输入 latent、scheduler版本、dtype/device、归一化是否与部署一致。
- loss公式及权重、schedule-aware监督、递归/多步 rollout 训练、teacher forcing 与自回归输入的差别；优化器、LR、batch、updates、数据量、seed与训练成本。论文和代码不一致时分别列。
- 接受/回退机制和分布假设：可计算的密度比、耦合噪声、误差代理、最大合法前缀或refresh分别如何实现；理论结论约束什么对象，哪些约束不能迁移到本项目。
- 闭环 SR、环境步/trace定义、失败、horizon、配对/区间、chunk/task/E2E边界；缺项写未报告。
- 可复用文件/函数的固定链接，未开源部分、专用 checkpoint或大模型重训依赖。

不要把连续 Flow 的确定性误差阈值、DDPM概率接受和离散 token 的分布保持混为一谈。即使论文证明去噪 latent trajectory 偏差受限或采样分布保持，也不能直接推出机器人成功率不降、环境轨迹长度不变。

## 4. 对本项目给出有证据的优先级

请回答：现在是该先查 teacher 的 BF16 B2/B14 数值与接口一致性，还是条件缺失、递归访问分布、训练标签不匹配，或接受代理与环境风险失配？不允许仅凭 TF 失败或 CAL更低指定根因。先把每种解释的支持、反证、尚缺字段列明；缺原始 teacher tensor 时不能声称已重算残差。

按 teacher自身→draft→接受路径→scheduler→保存动作/原后处理→提交/环境命令→失败分层。teacher先有差异，说明怎样影响 label/接受，如何检查且保留旧结果。命令组件审计不能解释接触失败。

无 cross-attention 是否造成不可辨识？需什么同 state/时间/previous velocity、不同 target方向的证据？复用 target特征与新 encoder各多少成本？不直接推荐加宽/层/图像。

只推荐**一条主线＋一个有明确触发条件的备选**。主线先完成有限诊断，再决定是否值得申请额外训练；已用完的训练额度不能自动重置。候选扩容、seed sweep、逐task阈值、无限新采集、PF2/3/4搜索均不接受。若判断当前证据已足够 NO-GO，应明确停止 TF 调优的理由及如何保留它为 exploratory，而不是保证一定训到过门。

## 5. 输出可交给 Codex 的有限执行方案

每个实验列问题/假设、固定输入、唯一改动、对照/字段、CPU核/GPU小时/墙钟/存储/完整episode上限、验收/停止/清理。replay不冒充闭环，禁止截断/smoke；未知耗时给范围及测量法。

是否值得恢复固定 DEV 剩余24局，补什么信息？不得替换失败、混旧 T/新 fresh作 TEST。risk已失败，补齐能否只完善描述而不满足旧门？新协议需前瞻登记、另授权。

能否上大规模训练/TEST？列最少缺口。若两卡获授权，只独立完整 episode分片并行，保留配对/初态/RNG/日志；53k draft不做 DDP，跨卡计时不混写。

最后给从恢复授权起的乐观/常规/不利时间表：尽快固定 draft 结构、训练方法和权重；列何时停止调优，何时仅保留系统机制研究，哪些证据允许开始独立 TEST。对齐10/22 go/no-go、10/23冻结、10/28主表、11/01 18:00北京时间给老师的中文证据包；过期就收缩范围，不承诺投稿或录用。不得用旧日历自动授权 GPU。

输出顺序：①固定版本与事实核对；②3–5篇证据卡与可迁移性表；③当前机制诊断优先级及缺证；④一条主线和一个触发备选的最小改动；⑤有限预算/停止门/时间表；⑥给 Codex 的执行清单，标清哪些仅为建议、哪些需用户另行授权。每个关键事实紧邻原文或固定代码链接；UNKNOWN 保留，不用“应该有效”替代结果。
