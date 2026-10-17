# 本轮 1→2→3 的实际接续入口

> **10/17早用户已暂停，GPU已清理。** 最新72/96终态、三新增风险失败及24缺项见[RESULT](RESULT.md)。下文为带时间的原运行历史，不是恢复授权；不要按旧“当前/接续”启动作业。

本页是运行接续，不是新授权。主登记：`投机采样/results/runtime/p0-tf-effect-closure-v1-20261016-r1/PREREGISTRATION.json`。科学基线 b0dfac3a，原质量四false、FAILED、默认和tile保持。不得修改正在运行collection manifest中冻结的源码。AutoDL不连接，120/TEST不自动启动。

用户追加要求（10/16晚）：完成后统一评估AutoDL单机双卡能否加快研究，给出质量/采样吞吐、最终成功率确认与仅投机E2E正式计时的分工、资源和工期；现在不单独回答。属于最终报告需求，不是租卡、旧AutoDL重连或绕过独立TEST/性能质量门的授权。

## 当前（2026-10-17 08:16，北京）

1. 诊断已完成，触发一次新数据实验；明确旧mixed反证及单失败DEV的因果限制，见EXPERIMENT。
2. 资格r1监督准入FAILED/0GPU；r2重复OFF/ON逐请求KV因开销主动PARTIAL，保留；r3复用原完整OFF、ON完整50/101请求/2192forward/904树/2712行逐位核验通过，RESET/final全KV及逐chunk RNG一致。没有完成原双新pass/allrequest KV合同。三Attempt均在原21:49:55帽内结束清理。
3. 完整CPU资格`p0-tf-effect-collector-reuse-audit-cpu-v1-20261016-r1/RESULT.json` PASS，23.8秒。扩展初态0–37向量审计PASS；两个先前CPU格式失败保留。首条完整新episode数据通路审计PASS，975全node/607reached，norm误差1.49e-8，不能代替30局总门。
4. **30完整采集已结束并清理**：`p0-tf-effect-collect30-v1-20261016-r1`，实际9383.734秒、23685个target forward、约132.3MiB；root3672051/clock3672054及全部观察到的自有worker已退出。作业COMPLETE不等于模型验收。
5. **全量CPU pack已PASS**：`p0-tf-effect-collect30-pack-cpu-v1-20261016-r1/DATA-AUDIT.json`，49.367秒，9797树/29391全node，18221实际到达行（FIT11416/CAL6805）、11170反事实保留，0额外teacher、0 CPU重算接受翻转；生成池29成功/1完整失败不作为新TF质量。`columns.pt`是后续训练输入。
6. **固定三条件训练与完整CPU1校验已PASS并清理**：`p0-tf-effect-onpolicy-train-cpu1-v2-20261017-r1`，1061.888s、约73.2MiB、11源不变；CONT复用先前500、新增0，ON/MLP首次各500；累计旧+新CPU作业2827.053s。三条件旧20237+新6805完整校验通过，不继承CPU2结果，生产GPU CAL仍在各arm开始前执行。旧r1 FAILED保留。
7. **DEV96已真实启动**：`p0-tf-effect-dev96-v1-20261017-r1`，root3787524/clock3787527，4臂fresh80+risk16、7h/3GiB，原08:06:37停工、08:11:37清理不变。新训练root和原报告源码已冻结；用户最新10:00上限及缺项接续条件见下。无120/独立TEST/正式性能。

06:24动态检查：完成51/96（T28、TF_CONT20已结束并各自清理，TF_ON3/28完整终态），59483个完整落盘target forward、约90.9MiB。TF_ON新GPU CAL已实际PASS：旧20237＋新6805共27042行同设备BF16逐位一致，调用179.186秒、0 synthetic/target forward，CPU/CUDA RNG不变；READY/O3/setup RNG恢复为true。TF_CONT复用原完整20237行证据，实际新增CAL执行0。MLP_ON尚未开始；独立clock仍在原预算内，没有接续作业。进度/接口PASS不等于机器人质量门判定。

07:38固定fresh20原记录进度读数（尚非完整状态/命令/初态审计）：T20/20，TF_CONT19/20（task7/init37完整失败796步），TF_ON20/20；TF_ON与T共同成功20组步数比P50/P90/max为0.990745/1.020326/1.027027，实际总步数T5311、CONT5779、ON5222，失败800均值265.55/289.15/261.10。开始TF_ON的8条风险回归；MLP仍未启动。固定权重/阈值不改，此时不推荐TEST、不签PASS；原四权限保持false。原client总墙钟T6712.99s/CONT4907.40s/ON4387.34s仅用于资源预估，不作正式speedup。

07:48风险首局原记录已出现新增失败：task1/init29，T264步成功，TF_ON完整horizon失败、实际796步，client615.990秒。严格初态/命令/相机审计仍待终态后执行；不得将fresh20/20写作整体PASS，不能放行独立TEST。继续固定余项及清理，不新增训练、接受器或阈值搜索。此反例与旧task4失败分别保留，不能互相覆盖。

用户01:31最新明确改为**10/17 10:00**，已Maintain：`memory/global/requirements/decision-本轮TF验证截止与晨报时间.md`，预算来源`MASTER/USER-1000-DEADLINE-AMENDMENT.json`。原DEV守原帽终止并清理：70/96完整机器人终态，SUPERVISOR/CLOCK为FAILED_OR_PARTIAL，24893.568秒，76533已落盘forward；旧TF_ON task4/init26未完整、中断forward实际UNKNOWN，保守扣4000。全部旧自有root/clock/worker退出，原FAILED不改。

**一次缺项接续已经登记并实际启动**：`p0-tf-effect-dev96-continue-v2-20261017-r1`，root4005474/clock4005493，08:08启动，pending26＝TF_ON风险6＋MLP_ON fresh20。原70个完整成败不重跑；09:35停工/09:40 GPU清理/09:55 CPU/10:00汇报。累计forward280000/3GiB不重置，新可用forward199467、空间3080688815字节。实际launch索引`MASTER/CONTINUATION-ACTUAL-LAUNCH-20261017.json`。08:13新TF_ON完整27042行GPU CAL已PASS，当前task4/init26完整任务；新完成0/26，不能说96完整。新5份源码与原204冻结源核验见`MASTER/CONTINUATION-SOURCE-REVIEW-20261017-r2.json`。

原TF_ON风险task3/init26也完整失败796步（T240成功）；与task1/init29一起保留，不能以fresh20/20遮蔽。原TF_ON臂因SIGTERM落在finally，缺`arms/TF_ON/OWNED-CLEANUP.json`；全局FINAL-OWNED/CLOCK证明全部自有生命周期退出，但不能伪造局部收据或把22条旧TF_ON记严格审计PASS。独立双namespace汇总保留该PARTIAL；另准备仅四条既有完整记录的CPU组件诊断，核实际命令/物理/RNG/相机，不能替代旧门。MLP/风险余项、严格审计及最终NO-GO待实际终态，不新增训练/阈值/TEST。

08:16 CPU组件诊断已实际完成：`p0-tf-effect-recorded-risk-audit-v1-20261017-r1/RESULT.json`，运行12.431秒（准备/等待/执行18.091秒）、CPU单核、0模型forward/0CUDA/0新episode。task1/init29与task3/init26四条完整任务命令/保存动作核验通过，两对实际初始sim/controller、完整reset、CPU/CUDA RNG字节和双相机均exact；当前TF_ON的两条新增风险失败真实保留。结果仅`COMPLETE_RECORDED_COMPONENT_DIAGNOSTIC`，不补原局部清理收据、不签质量证书、不推断失败根因。即使fresh和后续MLP条件通过，当前配方也不能满足事前无新增风险失败门。

若确需接续，先读`投机采样/experiments/tf_effect_campaign_continue_v2/README.md`，固定新job `p0-tf-effect-dev96-continue-v2-20261017-r1`。其plan保留原96，`pending_keys`只含M缺项；新终态只签M局。旧完整成败缺关闭审计则保留并使总门PARTIAL，不能重跑替换。接续后读`tf_effect_completion_report_v2/README.md`，用独立report namespace跨两个真实来源审计，不套用原单job96完整PASS入口。普通STOP/模型错误不触发接续；只接纳经精确deadline时钟证明的原预算终止。全部新CAL、setup、中断forward保守扣账，原FAILED和原权限均保持。

原先用户睡前要求09:00汇报，已在DEV开始前新增`MASTER/USER-0900-DEADLINE-AMENDMENT.json`：未来DEV清理≤08:40、CPU分析≤08:55、汇报≤09:00，同时取原7h/12h及任何更早帽。旧登记和原CPU1800秒帽不改；未来独立prepare绑定新修正。AutoDL旧实例不连接，双卡仅评估建议。

00:52原CPU job已FAILED并清理，实际1765.165秒、CONT500、ON/MLP0；所有9冻结源未改。严格自有身份校验后SIGINT使原handler在外层TERM前留FAILED，不覆盖失败。00:53独立`tf_effect_train_cpu_reuse_v2/train.py`新job `p0-tf-effect-onpolicy-train-cpu1-v2-20261017-r1`准备15.262秒通过并实际运行：复用CONT500、不重训，ON/MLP仍首次各500，训练2线程、完整逐行校验1线程，GPU CAL门不减。新1800秒/256MiB，原耗时累计；不自动第二次续接。见`MASTER/CPU-INFRASTRUCTURE-CONTINUATION-PLAN.json`及新job `ARTIFACT-REUSE-PROOF.json`。

22:38接线复审：补尚未使用的DEV launcher回执`start_new_session=true`并绑定其源码SHA；报告CPU deadline取全轮global hard之min，真实保存RNG张量的shape/dtype/bytes绑定READY（本机真实档案CPU核对通过、0模型forward/0CUDA）。独立agent只读复核顺序、CAL/训练、动作后处理、phase/终态未发现其他确定阻塞。未来DEV prepare一并冻结最终报告器源码，不能看结果后改计算规则。当前collection冻结源码全量无变化，采集10/30；未改变生产算法、数值或质量门。Cycle表补齐M12/M13，不涉及保护state/cycle/rules。

## 接续，按顺序执行

所有CPU命令用`CUDA_VISIBLE_DEVICES=''`、`OMP/MKL/OPENBLAS_NUM_THREADS=2`、`PYTHONDONTWRITEBYTECODE=1`，`taskset -c 12,13`及`/rjs/cxz/miniconda3/envs/va/bin/python -B`。不运行smoke或截断episode。

以下`experiments/...`相对路径均以`投机采样/`为根；runtime路径使用上文完整目录，不混用旧namespace。

1. 等采集`COLLECTION/SUPERVISOR/CLOCK-RESULT/FINAL-OWNED-CLEANUP`真终态。若失败/不足30，保留PARTIAL，不凑数据训练；检查真实原因，不重置原4h帽。训练门尚不成立。
2. 完整pack：`experiments/tf_effect_collector_audit_v1/pack.py --prepare --job NEW_PACK --collection COLLECT_JOB`，随后`--run --job NEW_PACK`。建议namespace `p0-tf-effect-collect30-pack-cpu-v1-20261016-r1`；900秒/1GiB。输出`columns.pt`及`DATA-AUDIT.json`，真实20FIT/10CAL root+reached；先原字段/score/物理命令/全局树数与资源清理审计。
3. 原训练r1已FAILED，不重跑。等待当前`p0-tf-effect-onpolicy-train-cpu1-v2-20261017-r1`实际终态；入口`tf_effect_train_cpu_reuse_v2/train.py`保持原v1 schema/API。完整PASS后才DEV；未来DEV的`--training`必须指此新root，不指原FAILED。TF_CONT/TF_ON/MLP_ON均seed23、child2500→500、同norm/loss/draw；CONT复用已500参数并与原旧CONT全tensor/norm一致，实际新增更新0，ON/MLP首次各500。完整CPU1校验不声称CPU2逐位继承；MLP只对照，不选架构/seed；不自动再次续接。
4. 未来DEV原5h预测在**新训练/DEV开始前**按首条270.5秒任务改为最多7h，但全轮12h绝对截止不变；证据`MASTER/DEV-BUDGET-FORECAST-AMENDMENT.json`。用独立准备入口`tf_effect_dev_budget_v2/prepare.py --job NEW_DEV --audit QUAL_CPU/RESULT.json --training TRAIN_JOB --trigger MASTER/TRAINING-TRIGGER.json`，它仅改未来cap常数并绑定新来源，原producer不动。建议`p0-tf-effect-dev96-v1-20261016-r1`。
5. 实际启动已登记DEV：`CUDA_VISIBLE_DEVICES='' /usr/bin/python3 experiments/tf_effect_job_launch_v1/launch.py --job NEW_DEV`。它启动原campaign run与独立clock，先等clock attach。新ON的旧20237+新CAL在对应arm开始前临时prepare，并在owned server执行GPU逐query接口验收；不能提前几小时prepare致900秒过期。顺序T/CONT/ON/MLP_ON，共fresh80+risk16=96完整局。无120或独立TEST。
6. DEV真终态后：`tf_effect_quality_report_v1/report.py --prepare --job DEV_JOB --out NEW_REPORT`，立即`--run --out NEW_REPORT`；建议`p0-tf-effect-dev96-report-cpu-v1-20261016-r1`，900秒/128MiB。读取README的精确接口，不把`--job`误作report输出。
7. 汇总三步实际结果和清理，更新HW；五报告由report生成。补总体预算与所有前驱FAILED、校验源/权重/保护491项、保留raw证据索引。四旧权限永false，性能ΣT_round/Σg未采为UNKNOWN；仅在新DEV的SR/length/risk及初态配对门都过后推荐下一阶段独立TEST，不自动启动。

## 终止与当前限制

旧master12h上限10/17 09:04:55和旧09:00修订记录均保留；最新10:00用户授权只通过独立缺项接续使用，不能延长原活跃作业的真实plan或静默重置旧耗时。接续如需运行，09:35停工、09:40清理，汇总09:55、用户10:00截止。不是AutoDL续租。GPU时间报作业生命期/墙钟，实际CUDA active UNKNOWN。新任务每局horizon800，失败记800但实际步数可796，不能互换。共享资源耗时不作正式加速比。原tile/融合不动；无本地论文初稿。
