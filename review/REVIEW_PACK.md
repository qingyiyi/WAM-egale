# 单文件阅读包：paused72

以本文件所在新commit为准；原b0dfac3a不代表最新。完整源码请依相对链接另读，缺原张量不声称复验。

# 新版TF draft审阅入口：暂停72局

当前snapshot_id：`wam-tf-draft-handoff-20261017-paused72`。固定本文件所在的新提交，旧科学基线`b0dfac3a02f65de33a1217246b83003615166640`只用于历史；不要随main更新。

用户已暂停GPU，先请网页版查原文与官方开源，为TF结构/训练给出一条有限且可证伪主线。暂停不是新的训练或TEST授权。

## 最新结果

- 一次30完整新数据、TF-cont/TF-on/MLP-on公平500更新完成。数据来自旧TF_CONT3000，不是新TF-on自己的on-policy。
- 固定96完成72条机器人终态，24未完成；50条逐局必要组件审计完整，22旧TF-on arm清理收据缺失，总体严格证书PARTIAL。全局清理证据不补造局部收据；暂停不是预算completion，原issue保留。
- fresh20：T20/20、CONT19/20、TF-on20/20；MLP-on0/20。TF-on/T共同成功步数P50/P90/max=.990745/1.020326/1.027027，仅描述，不签严格PASS。
- TF-on风险已完整4/8，0/4成功；task1/29、3/26、4/26 T成功/TF失败，task5/28双方失败。三条新增失败均有初始物理/reset/RNG/双相机exact证据。当前配方不应推荐独立TEST；TF架构不是已证伪。
- 四项权限仍false；旧A_off learned draft默认未改。120、独立TEST、新TF正式45chunk/新系统实验未启动，ΣT_round/Σg和新TF正式E2E UNKNOWN。

## 先读这些

1. [完整版文献/代码调研Prompt](TF_DRAFT_RESEARCH_PROMPT.md)
2. [结果JSON](RESULTS.json)、[本轮总览与五报告](../.pipeline/experiments/tf-effect-closure-20261016/RESULT.md)、[短HW](../.pipeline/PROGRESS.compact.md)
3. [TF/MLP结构](../投机采样/experiments/draft_architecture_v1/models.py)、[固定训练](../投机采样/experiments/tf_effect_train_v1/train.py)、[CPU恢复与复用](../投机采样/experiments/tf_effect_train_cpu_reuse_v2/train.py)、[teacher采集](../投机采样/experiments/tf_effect_collector_v1/collector.py)
4. [接受和合法提交](../投机采样/scripts/p0_batched_verify.py)、[真实target](../投机采样/wam-p0-worktree/wan_va/modules/model.py)
5. [单文件阅读包](REVIEW_PACK.md)、[增量来源manifest](SOURCE_MANIFEST.json)、[遗漏原件](OMISSIONS-20261017.md)。

原始PT/NPY/相机/KV/模型权重不公开，不能声称仅凭GitHub已重算所有原始张量。服务器绝对路径是历史证据引用；在仓库中去掉`/rjs/cxz/论文探索/`前缀映射相对路径。


---
# 本轮总览

# TF64唯一一次数据与训练收口：暂停交接结果

2026-10-17早，用户明确暂停，先做网页版文献/开源复核并同步GitHub。本页是派生总览，原FAILED、旧retention和四权限全部保留。科学基线`b0dfac3a02f65de33a1217246b83003615166640`；新快照不能覆盖旧版本判断。

## 结论

TF64结构/API可继续暂冻作系统研究载体；**当前TF-on权重/配方不满足无新增风险失败条件，不能推荐独立TEST或正式性能**。原固定96仅72条完整机器人终态，剩24未完成；matched MLP-on闭环为0/20，不能宣布TF优于MLP或改选MLP。原严格总证书为PARTIAL，不把局部改善改成PASS。

## 三步实际产物

1. CPU监督/联合覆盖诊断57.14秒，支持唯一一次当前访问分布对照；旧mixed反证保留，不认定严重偏移或失败根因。
2. 一次30完整episode（20FIT/10CAL）已采集并清理；9383.734秒、23685 target forwards、额外teacher0。生成策略为旧TF_CONT3000，生成池29成功/1失败不是新TF-on质量。全量pack：FIT11416/CAL6805实际到达行，11170反事实只存原档。
3. TF-cont/TF-on/MLP-on公平500更新均完成。原CPU校验job1765.165秒FAILED保留；CPU1完整校验续接1061.888秒PASS；CONT复用已完成权重，ON/MLP首次各500。normalization/seed/loss/阈值/生产API固定。TF-on新CAL scaled-L2均值.003567824，相对CONT .003734550低约4.46%；逐query通过90.389%对90.066%，不等于任务SR。TF-on两次实际完整27042行GPU接口核验通过；MLP生产GPU CAL未运行。

## 完整机器人终态描述（不同分层不混池）

|分层|方案|已完成/计划|成功/已完成|实际环境步总和|失败800均值|
|---|---|---:|---:|---:|---:|
|fresh init36/37|T|20/20|20/20|5311|265.55|
|fresh init36/37|TF-cont|20/20|19/20|5779|289.15|
|fresh init36/37|TF-on|20/20|20/20|5222|261.10|
|fresh init36/37|MLP-on|0/20|UNKNOWN|0|UNKNOWN|
|known-risk|T|8/8|6/8|3244|406.50|
|known-risk|TF-on|4/8|0/4|3184|800.00|

TF-on/T fresh共同成功20条原终态步数比P50/P90/max=.990745/1.020326/1.027027。fresh TF-on20条仍带原arm清理收据缺口，这些只是描述值，不能称严格质量PASS。CONT失败task7/init37完整796步。

|task/init|T|TF-on|初始配对与命令证据|
|---|---|---|---|
|1/29|264步成功|796步完整失败|独立四局组件诊断，物理/controller/reset/实际CPU-CUDA RNG/双相机exact|
|3/26|240步成功|796步完整失败|同上，全部保存动作及环境命令核验|
|4/26|249步成功|796步完整失败|新接续局完整审计、跨namespace初态/RNG/双相机exact|
|5/28|796步失败|796步完整失败|同上；双方失败不能抵消前三对新增失败|

## 审计、暂停与放行边界

原DEV 70完整后按原预算FAILED_OR_PARTIAL；SIGTERM落在finally，缺旧TF_ON局部arm cleanup，22条完整成败均保留，不重跑替换。全局CLOCK/FINAL-OWNED证明自有生命周期退出，不能补造旧局部收据。

一次缺项接续原pending26；实际只完成task4/task5两条后，用户要求暂停。STOP在完整episode_terminal阶段记录；下一case已生成client启动元数据，但没有完整episode终态或COUNTER，随后guard拒绝并清理；不得称该任务已运行完成或算法失败。该未闭合attempt的实际forward UNKNOWN，另按原每局4000上界扣账；接续原终态FAILED_OR_PARTIAL保留。独立clock清理确认60个已观察自有pins全退出。不是补齐96，也没有在用户暂停后继续跑余项。

双namespace只读CPU报告：72条机器人终态、50条必要审计完整、22条旧arm缺口；用户暂停属于非预算completion而额外列source_terminal issue，报告严格保持PARTIAL/NO_SELECTION。不修改冻结汇总器以接纳暂停或改PASS。两对旧风险的独立组件诊断与新风险审计共同提供三条T胜/TF败反例；这些反例足以阻止推荐当前配方，并不把总体UNKNOWN证书改成已完成。

剩24：TF-on风险task6/init27、task8/init25、task0/init28、task6/init25；MLP-on fresh20。没有120、独立TEST或正式45chunk。ΣT_round/Σg、最终投机E2E加速、独立非劣均UNKNOWN。旧A_off默认不改。

## 原始证据与五报告

- [双namespace机器结果](../投机采样/results/runtime/p0-tf-effect-dev96-completion-report-cpu-v2-20261017-r1/RESULT.json)
- [FACTS](../投机采样/results/runtime/p0-tf-effect-dev96-completion-report-cpu-v2-20261017-r1/FACTS.md)
- [QUALITY-RESULT](../投机采样/results/runtime/p0-tf-effect-dev96-completion-report-cpu-v2-20261017-r1/QUALITY-RESULT.md)
- [SYSTEM-RESULT](../投机采样/results/runtime/p0-tf-effect-dev96-completion-report-cpu-v2-20261017-r1/SYSTEM-RESULT.md)
- [CLAIM-EVIDENCE-MATRIX](../投机采样/results/runtime/p0-tf-effect-dev96-completion-report-cpu-v2-20261017-r1/CLAIM-EVIDENCE-MATRIX.md)
- [STOP-REPORT](../投机采样/results/runtime/p0-tf-effect-dev96-completion-report-cpu-v2-20261017-r1/STOP-REPORT.md)
- [两对既有风险组件核验](../投机采样/results/runtime/p0-tf-effect-recorded-risk-audit-v1-20261017-r1/RESULT.json)
- [全轮资源账本](../投机采样/results/runtime/p0-tf-effect-closure-v1-20261016-r1/FINAL-RESOURCE-LEDGER-20261017-PAUSED-r2.json)
- [训练来源与实际接续](../.pipeline/experiments/tf-effect-closure-20261016/RUNBOOK.md)；[文献机制边界](../.pipeline/research/speculative-sampling/TF-RECIPE-LITERATURE-RECHECK-20261017.md)

## 下一步（不自动执行）

当前所有GPU实验暂停；用户明确继续后才另登记新作业，不能清除旧STOP/FAILED或重置接续预算。优先请网页版对原文/官方代码核验draft条件、监督目标和连续接受机制，给一条有限且可证伪主线，避免盲目大训练。本轮唯一新数据/公平训练机会已用完，不继续加轮或architecture sweep。TF结构可暂冻；权重/训练法不可签最终合格。

AutoDL双卡可把独立完整episode分配到两张同型号A100≥40GiB、总≥16CPU/64GiB RAM，无需53k draft DDP；实际双卡吞吐UNKNOWN，不能保证2倍，且不能修复质量。完整SR/单chunk/task/E2E仍须候选先过门，正式速度回本机同SXM4独占测量。当前未重连或租用AutoDL。

10/18作者profile，10/22质量/系统go-no-go，10/23冻结只在证据允许时，10/28主表，11/01 18:00中文证据与稿件给老师。质量未过不预排必然启动TEST，不保证原完整主张按期成立。


---
# 完整检索与规划Prompt

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


---
# TF结构原代码

```python

"""Isolated, CPU-testable two-proposal architectures; production sources unchanged."""
from __future__ import annotations

import math
from pathlib import Path
import torch
from torch import nn

SPECS = {
    'mlp51': {'kind': 'mlp', 'hidden': 51},
    'mlp96': {'kind': 'mlp', 'hidden': 96},
    'transformer64': {'kind': 'transformer', 'width': 64, 'heads': 4, 'ffn': 256,
                      'layers': 1, 'dropout': 0., 'tokens': 16, 'activation': 'gelu', 'norm_first': True},
}


class ResidualMLP(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(227, hidden), nn.SiLU(),
                                   nn.Linear(hidden, hidden), nn.SiLU())
        self.head = nn.Linear(hidden, 224)

    def forward(self, x):
        center, delta = self.head(self.trunk(x)).reshape(-1, 2, 112).unbind(1)
        return torch.stack((center + delta, center - delta), 1)


def position_tokens(x):
    """Canonical [channel, frame, subaction] flattening -> 16 position tokens."""
    if x.ndim != 2 or x.shape[1] != 227:
        raise ValueError('expected [batch, 227] normalized features')
    state = x[:, :112].reshape(-1, 7, 16).transpose(1, 2)
    previous = x[:, 112:224].reshape(-1, 7, 16).transpose(1, 2)
    scalars = x[:, 224:].unsqueeze(1).expand(-1, 16, -1)
    return torch.cat((state, previous, scalars), -1)


class PositionTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Linear(17, 64)
        self.position = nn.Parameter(torch.empty(1, 16, 64))
        nn.init.normal_(self.position, std=.02)
        self.block = nn.TransformerEncoderLayer(64, 4, 256, dropout=0., activation='gelu',
                                                batch_first=True, norm_first=True)
        self.head = nn.Linear(64, 14)

    def forward(self, x):
        y = self.head(self.block(self.embed(position_tokens(x)) + self.position))
        center = y[:, :, :7].transpose(1, 2).reshape(-1, 112)
        delta = y[:, :, 7:].transpose(1, 2).reshape(-1, 112)
        return torch.stack((center + delta, center - delta), 1)


def make_model(name):
    spec = SPECS[name]
    return ResidualMLP(spec['hidden']) if spec['kind'] == 'mlp' else PositionTransformer()


class Predictor:
    def __init__(self, architecture, normalization):
        self.architecture = architecture
        self.model = make_model(architecture).float()
        for key in ('feature_mean', 'feature_std', 'residual_mean', 'residual_std'):
            setattr(self, key, normalization[key].detach().clone().float())
        if self.feature_mean.shape != (227,) or self.residual_mean.shape != (112,):
            raise ValueError('normalizer shape mismatch')
        if any(not bool(torch.isfinite(getattr(self, k)).all()) for k in normalization):
            raise ValueError('nonfinite normalizer')
        if bool((self.feature_std <= 0).any()) or bool((self.residual_std <= 0).any()):
            raise ValueError('normalizer must be positive')

    def decode_batch(self, data):
        features = torch.cat((data['state'], data['previous'], data['scalars']), 1)
        normalized = (features - self.feature_mean) / self.feature_std
        raw = self.model(normalized)
        return (data['previous'][:, None] + raw * self.residual_std[None, None]
                + self.residual_mean[None, None])

    @torch.no_grad()
    def predict(self, current_state, previous_direction, sigma, next_sigma, timestep):
        if current_state.shape != previous_direction.shape or current_state.numel() != 112:
            raise ValueError('two-head predictor requires matching compact 112-element tensors')
        if current_state.device.type != 'cpu' or previous_direction.device.type != 'cpu':
            raise ValueError('this experiment only authorizes CPU prediction')
        if not all(math.isfinite(float(v)) for v in (sigma, next_sigma, timestep)):
            raise ValueError('nonfinite time scalar')
        if not bool(torch.isfinite(current_state).all()) or not bool(torch.isfinite(previous_direction).all()):
            raise ValueError('nonfinite input')
        self.model.eval()
        data = {'state': current_state.detach().reshape(1, 112).float(),
                'previous': previous_direction.detach().reshape(1, 112).float(),
                'scalars': torch.tensor([[sigma, next_sigma, timestep / 1000.]], dtype=torch.float32)}
        value = self.decode_batch(data).reshape(2, *current_state.shape).to(current_state.dtype)
        if not bool(torch.isfinite(value).all()):
            raise ValueError('nonfinite output')
        return value

    def save(self, path, metadata):
        torch.save({'schema': 'draft-architecture-cpu/v1', 'architecture': self.architecture,
                    'specification': SPECS[self.architecture], 'metadata': metadata,
                    'model_state_dict': self.model.state_dict(),
                    'normalization': {k: getattr(self, k) for k in
                                      ('feature_mean', 'feature_std', 'residual_mean', 'residual_std')}}, path)

    @classmethod
    def load(cls, path):
        data = torch.load(Path(path), map_location='cpu', weights_only=False)
        if data['schema'] != 'draft-architecture-cpu/v1' or data['specification'] != SPECS[data['architecture']]:
            raise ValueError('checkpoint architecture mismatch')
        result = cls(data['architecture'], data['normalization'])
        result.model.load_state_dict(data['model_state_dict'], strict=True)
        result.model.eval()
        return result


```
