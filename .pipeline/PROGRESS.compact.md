# 当前研究恢复入口

更新时间：2026-10-17 14:02，北京。派生短入口；原FAILED/tile/协议保留。先读[INDEX](INDEX.md)、[上下文策略](CONTEXT-POLICY.md)、[约束记忆](memory/INDEX.md)，再恢复当前Cycle，不展开全部旧知识。

## 当前M16：全量描述评估准备

用户新增授权Long与LIBERO90分别Target/TF/TF+PF2全量、90另训draft、现有AutoDL双卡可用至完成。见[计划](experiments/libero-fullscale-20261017/PLAN.md)、[授权](memory/global/requirements/decision-AutoDL全量研究使用至完成.md)。新Long登记500组（10task×50init）×三臂1500完整局；init-major任务均衡，同case同GPU，T_C_P/C_P_T/P_T_C轮换。每卡单持久server，server/client各2CPU；源码/有界fixture准备中，尚未启动新GPU。新job48h含启动/等待/失败/清理，10/19 13:40:56停工作、13:55:56清理止，10GiB证据含fixture3GiB、磁盘留5GiB。旧M15起止不改，不自动重试。

已检索官方与68个同名HF模型，尚未核实同代LingBot-VA LIBERO90 target；社区Wjjjh有Object/Goal/Spatial各10task模型、不是90；rbehnam19六权重无suite/normalizer/评测身份；FastWAM90是另一架构。独立subagent继续深挖一手来源。90采集/训练/GPU均未启动，不能把Long权重当已适配90。

用户同意[质量并行、保存真实state后独占计时](memory/global/requirements/decision-质量并行评估与独占重放计时分离.md)。现有actions/latents不够完整replay；只补短中长三份request前完整已知state+输入+少量oracle，总≤3GiB，不截episode或多调模型。GPU重放可测实际边界的verifier/chunk，不能替代完整任务E2E。并行时间仅探索吞吐；同卡多eval须实际显存与独立KV/RNG/scheduler支持，不能简单并发复用可变服务。

自然模型失败不阻止新全量覆盖；raw初态/首两S不等保留FAILED并标严格pair INVALID。非法route/API/数学合同/资源硬限停新工作。官方50init含已用FIT/CAL/DEV与原保留组，不将新描述benchmark签为独立TEST。预计Long双卡25–40h，需实际完整任务吞吐更新；90工期在Target未核前UNKNOWN。

四权限恒false：engineering_quality_screen、independent_test_complete、noninferiority_established、formal_performance_permission。旧TF_ON/PF2 NO_GO、retention、22清理缺证/FAILED及全局A_off保持；未启动120、独立TEST、新TF正式45chunk或正式E2E。

## M15已结束：六对十二完整DEV，投入NO_GO

[原结果/五报告](experiments/tf-pf2-autodl-20261017/RESULT.md)。C3/6、P5/6，rescued2/lost0；风险0/3→2/3，对照3/3→3/3。1/29与3/26救回，4/26两败；控制1/36为260→405。共同成功长度P50/P90/max=1.0043/1.4470/1.5577；风险全成功、控制罚分、P90/max门失败。全体实际C3110/P2166，800罚分3122/2170；风险罚分2400/1305、对照722/865。实际796不补造800命令。

探索ΣT_round/Σg .04068/.05251秒（P/C1.2909），共同成功policy165.90/226.99秒。容器GPU PID未全映射，成本可比UNKNOWN、不是Target或正式E2E。六对保存的物理/controller/RNG/双相机组件一致，首两S现场有效KV/RNG比较；reset/S0仅排除native覆盖六lazy scheduler字段，不是全部隐藏状态/离线全KV证书。首两S score/depth不能清晰区分风险，不能拟合阈值。

实际源码9acc37cee9ae9b62b8311f1f8dbb01bffa37a86a，父科学4f04。r1导入失败/r2两client零步失败保留；r3 runner20.89分钟，从原授权到终态1.962h，server累计.728764 GPUh非CUDAactive。1267原件52.5MiB已回传，owned组EMPTY/guard CLEAN、85身份无存活、双卡空。实例未关机，新使用为M16授权。249MiB临时资产包/20.1MB证据包双端删，原件保留。

原M15起点11:11:41、22:56停新/23:11清理为历史；12局/49k/3GiB不扩。计16054forward，r2缺COUNTER保守8000、非episode留1000，保守25054/49000；全局active/非episode真实计数UNKNOWN。新M16单独账本，不借旧余额。

## 固定快照、模型与前次证据

公开4f04b35e44cbc00483dc6245ad57abec718385c9 / wam-tf-draft-handoff-20261017-paused72永久72/96、MLP0、PAUSED_BY_USER_PARTIAL_NOT_QUALIFIED；live77/96另列，main/b0dfac3a不替代。新M15结果分支689a9d1，实际math9acc。见[77结果](experiments/tf-effect-closure-20261016/RESULT.md)：fresh T20、CONT19、ON20成功/各20；MLP1/1非计划完成，risk T6/8、ON4/8。三条T成功/TF败1/29、3/26、4/26，保存组件匹配、根因未定。旧22缺证、原暂停non-budget issue/FAILED不改，余19不自动补。

TF64 53070参数：16token×17feat(state7/previous7/σ,nextσ,t÷1000)，1层pre-norm D64/H4/FF256 GELU/drop0、学习位置、双7维残差头，无图像/prompt cross-attn；MLP96 52928仅baseline。根previous为已算target，递归previous父draft；内部FP32，生产BF16方向、FP32分数。

TF_ON seed20260923，child2500起点固定500更新/b512/lr3e-5/AdamW/wd.01，norm不变。新30由旧TF_CONT3000采9383.734s/23685forward；FIT11416/CAL6805 reached，11170反事实另留。旧target FIT159691/CAL20237，旧mixed14997=3906 A_off+11091历史TF。唯一Long训练机会已用完，90另授权但需可信teacher。真实AST主loss回归decoded raw velocity；std仅decode/gap，不是主MSE归一化或Δσ加权。27042行GPU CAL仅adapter、teacher0，不证明B2/B14一致。FIT无完整视觉/KV条件，227维表不能补造新latent teacher标签。

τ=.007839680526450805，absΔσ×L2，prefix_then_cost/K2B2/bonus冻结。PF2=S S T，成功policy返回计数/reset清零/cache不计，拒绝S仍计S；T canonical CFG2 action50更新/51调用含terminal，不是bonus或物理回滚。7状态×CFG2=B14，每attention仍Q16、私有KV不默认共享。

## 全项目运行边界与后续

[AutoDL不smoke/测试脚本/SHA](memory/global/requirements/requirement-AutoDL直接源码运行且不做smoke或SHA.md)包括隐式loader/fingerprint；必要审计inline真实完整任务。所有后续采样/质量/性能全臂[默认完整512-token BF16 reset bank](memory/global/requirements/decision-实验默认启用等价reset优化.md)，不能静默fallback；M15 reset15–107ms，首次构建/冷启动另列。

系统主线投机重组→可调度候选→超过最佳B14→质量约束有效成本。H1真实shape/H2批处理有证据，H3有限pilot，H4未立。30层D3072/H24×128/FF14336，Linear M32/96/224已合并，attentionQ16；G1→G2 L2−32.81%、DRAM−1.31%、occupancy降，不能写HBM减半。原tile只读，不恢复撤回融合；旧倍数不相乘。

旧系列GPU作业墙钟11.0265h、保守97039forward留历史。两旧下载残片回收2.38GiB（du137.25→134.87GiB）；canonical/原始失败/KV保留。HW固定v15/legacy，不升级/全量sync；保护state/cycle/rules、结构规则/491原tile记录不动。完整质量和正式性能仍分门，旧质量未通过不自动放行。

10/18作者profile仅记录；10/22 go/no-go；10/23冻结候选/API/claim；10/28主表；11/01 18:00中文证据包交老师。中文稿用户整理飞书、不建本地论文初稿；证据不足收缩claim。CVPR2027官方时间临近再核，未确认独立abstract规则。
