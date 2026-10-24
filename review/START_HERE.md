> 最新2026-10-24 FFN结果：[CURRENT-FFN-20261024.md](CURRENT-FFN-20261024.md)。四候选均无净收益、保留原A；先读此入口。旧快照身份/结果不改。

# 当前入口：RTX5090 Self QKV迁移（2026-10-24）

[最新算子结果与完整阶段表](CURRENT-RTX5090-SELF-QKV-20261024.md)。本轮所有对照均eager、无CUDA Graph；下方10/23的Graph说明仅属于旧实验。当前研究采用Self QKV TMA版本，四科学门及production_admitted仍false；旧TF暂停72等保留为历史，不能据它覆盖新的作业状态。

---

# 当前研究入口：局部warp有界实验完成（2026-10-23 16:34）

**计时口径：本轮 A/L1/L13/D/G 全部使用外层 CUDA Graph，自有算子也被捕获和重放；双方 eager 对照未测。** [说明及源码行号](../.pipeline/experiments/local-warp-pipeline-20261023/MEASUREMENT-MODES.md)。

请优先读[REVIEW_RETURN](../.pipeline/experiments/local-warp-pipeline-20261023/REVIEW_RETURN.md)、[完整阶段表](../.pipeline/experiments/local-warp-pipeline-20261023/STAGE-TABLE.md)、[最终同轮对照](../.pipeline/experiments/local-warp-pipeline-20261023/FINAL-COMPARISON.md)。源码固定提交1ef5b83af38d523b835f2511018334ba99a4b694；算子研究/optimized在此公开快照显示L13，但两机原运行入口未部署该版。

FFN2局部准备主线INCONCLUSIVE，不采用；LN1/LN3调制融合L13小幅改善0.4%–0.7%，中长KV仍慢强guarded约3.4%–3.6%。有限225调用/候选通过，不证明全部输入或机器人质量。四科学权限及production_admitted仍false。全部本轮GPU0 owned已退出，GPU1原RoboTwin继续；16:33配齐125组三臂，微波炉20/100组，等100组完整后才查因。原失败、旧预算起点/截止和retention不改。

这是新完成的有限算子实验，替代此前“等待网页版设计”的当前状态；下方TF暂停72是历史科学快照，不改其身份/结果。此前入口保留于[history](history/current-before-local-warp-20261023/START_HERE.md)。

---

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
5. [单文件阅读包](REVIEW_PACK.md)、[真实登记的有界派生摘要](REGISTERED_EXPERIMENT_SUMMARY.json)、[增量来源manifest](SOURCE_MANIFEST.json)、[遗漏原件](OMISSIONS-20261017.md)。

原始PT/NPY/相机/KV/模型权重不公开，不能声称仅凭GitHub已重算所有原始张量。服务器绝对路径是历史证据引用；在仓库中去掉`/rjs/cxz/论文探索/`前缀映射相对路径。

用户最新决定：等微波炉100组三臂完整后再查原因，原队列继续；[计划与CPU完成提醒](../.pipeline/experiments/robotwin-pilot-20261020/eval100-20261022/microwave-completion-20261023/PLAN.md)。
