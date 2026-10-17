# Long／LIBERO90全量三臂研究

本轮用户已明确要求执行Long与90的Target/TF/TF+PF2，并重新训练90 draft、双卡并行；[本轮决定](../../memory/global/requirements/decision-Long与LIBERO90全量三臂评估及独立draft训练.md)。旧M15六对结果与全部false不改，不继承其12局或49k作业余额。

- [x] 核对常用论文范围：常见四套各10任务×50次；LIBERO-90与100局不是同一概念。
- [x] 核对现有Target：Long官方模型、50局新reset基线和所有步数/计时原件保留；不是90/100任务基线。
- [x] 核对AutoDL资产/双GPU：13:29只读观察两卡空、原环境/Long模型/reset bank在，约17.1GiB可用；未启动新GPU。
- [ ] 独立实现Long三臂完整入口与预算守卫，静态审查后直接完整episode，无smoke/独立测试/SHA。
- [ ] Long 500组（task0–9、init0–49）×T/C/P共1500完整episode：按init外层/task内层均衡推进，两卡同case同卡、三顺序轮换。
- [ ] 核实90同代官方/社区Target；找到可信权重才登记90采集、独立TF64训练与同权重PF2对照，找不到报告具体BLOCKED。
- [ ] 全量成功/实际命令/800罚分/配对长度/ΣT_round÷Σg/policy/task/cold与warm成本、资源及清理，联合汇报。

Long拟定执行上限：每局1200秒/4000 target尝试；全1500局最多6,001,000尝试（含非episode1000），新增原始证据10GiB（含fixture≤3GiB）；每卡单持久server2CPU/client2CPU。不自动重试，不以更快失败作加速。所有初态、两相机、实际RNG与S/T路由inline审计。已见FIT/DEV与尚未使用初态分层，不签旧独立TEST或非劣证书。

用户已明确AutoDL用至工作完成，见[后续授权](../../memory/global/requirements/decision-AutoDL全量研究使用至完成.md)，不再将旧M15的23:11用于新job。Long新job生命周期上限48小时（两GPU并行，含启动/等待/失败/清理），工作截止比最终deadline提前15分钟；开case必须预留三个完整episode及清理。达到新cap报告PARTIAL，不自动重置或重试，不改旧账本。

用户同意[质量并行与独占计时分离](../../memory/global/requirements/decision-质量并行评估与独占重放计时分离.md)。现有actions/latents/commands与RAM首两S比较不足以完整离线重放，完整KV/数学条件/RNG持久fixture尚未实现；先直接完整质量运行，独立补最多3GiB短中长代表输入保存，避免记录所有query。保存成本单列，不将质量运行计时视作正式性能。非法API/route/预算停新工作；raw像素或首两S隔离不等保留FAILED并标严格pair INVALID，描述性覆盖继续，不伪造严格通过。

预计Long每臂100局的三臂工程覆盖约5–8小时；每臂500局完整标准规模约25–40小时，两A100并行、原环境保持且优化reset。是据已有PCIe完整任务外推，非计时实测；失败长尾/环境错误可能更久。90 Target身份未定，不提供虚假的完整训练/全量测试确定工期。初步量级与依据将独立归档。

下一步为独立源码冻结→登记实际job/source/清理身份→使用两卡推进Long；90文献与checkpoint核查并行。原tile/保护state/cycle/rules、旧FAILED及所有旧阈值保持。

实际Long登记：[plan.json](../../../投机采样/results/runtime/p0-tf-fullsuite-long-v1-20261017-r1/plan.json)，新起点2026-10-17 13:55:56、工作停10/19 13:40:56、清理止13:55:56。该48h上限不是旧M15重置。旧12局51.65MB原始episode外推1500约6.46GB，故加入≤3GiB fixture后将总cap从8改10GiB，仍保留磁盘5GiB底线。
