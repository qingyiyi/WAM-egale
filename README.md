# WAM-egale：WAM 投机与多候选 Transformer 研究快照

本仓库用于让研究者或网页版 ChatGPT **按代码与证据判断进展、风险和下一步**。快照时间为北京时间 **2026-10-16 下午**，最新完整质量验证为本机 50 条完整任务。

**当前结论：TF64结构和7状态/B14 verifier接口暂冻供算子研究；当前3000更新权重/训练配方未通过预登记保留条件。没有启动明天120局验证，四项质量/性能权限仍为false。**

## 从这里读

1. [评审入口与事实优先级](review/START_HERE.md)
2. [可直接复制的详细网页评审prompt](review/WEB_REVIEW_PROMPT.md)
3. [当前结果与状态JSON](review/RESULTS.json)
4. [单文件研究阅读包](review/REVIEW_PACK.md)：网页工具不方便逐文件读取时使用
5. [HW短状态](.pipeline/PROGRESS.compact.md)与[最新完整报告](.pipeline/research/speculative-sampling/DRAFT-FREEZE-RESULT-20261016.md)
6. [代码地图](投机采样/docs/CODE-MAP-20261014.md)、[逐文件来源清单](review/SOURCE_MANIFEST.json)与[未发布原件说明](review/OMISSIONS.md)

本页所在的Git提交就是评审版本。阅读和引用时固定该commit，避免后续main更新混入判断；不要把较旧的HW主题记忆当作当前运行状态。

## 最重要的结果

| fresh DEV：Long10/init35 | 成功数 | 平均环境步数 |
|---|---:|---:|
| Target | 10/10 | 257.5 |
| TF_CONT | 10/10 | 265.0 |
| 等更新MLP_CONT | 10/10 | 267.7 |

TF/Target共同成功长度比P50/P90/max为1.021212/1.074884/1.092593，三长度条件通过。

已知风险8组中双方各5/8，但task4/init26为Target249步成功、TF796步失败，触发原事前风险否决；该对初始保存物理、实际target Torch RNG及两相机全部一致。反向task0/init28的TF救回不抵消这项登记条件。风险组、fresh与target重复实验不得混成50个独立样本。

## 代码阅读顺序

- [真实target入口](投机采样/wam-p0-worktree/wan_va/wan_va_server.py)、[Transformer与KV实现](投机采样/wam-p0-worktree/wan_va/modules/model.py)
- [连续动作候选树/验证/接受与提交](投机采样/scripts/p0_batched_verify.py)、[当前policy桥接](投机采样/experiments/prefix_acceptance_v1/server_entry.py)
- [TF/MLP模型](投机采样/experiments/draft_architecture_v1/models.py)、[训练](投机采样/experiments/target_trace_draft_train_v1/train.py)、[混合微调](投机采样/experiments/target_trace_draft_finetune_v1/train.py)
- [本轮50局协议与分析](投机采样/experiments/draft_freeze_quality_v1/)、[实际预登记](投机采样/results/runtime/p0-draft-freeze-v1-20261016-r1/PREREGISTRATION.json)
- [跨候选attention原型](投机采样/experiments/wam_candidate_tile_v1/)、[guarded v2](投机采样/experiments/wam_candidate_tile_guarded_v2/)、[共同文本投影缓存pilot](投机采样/experiments/wam_static_prompt_baseline_v2/)
- [当前冻结研究接口](投机采样/results/runtime/p0-draft-freeze-v1-20261016-r1/OPERATOR-INTERFACE.STRUCTURE-FIXED.json)

## HW是什么

HW指Hypo-Workflow研究记录，位于[.pipeline/](.pipeline/INDEX.md)，保存实验协议、进度、证据指针、历史负结果和约束。这里是经过筛选的文本副本；不是全部服务器状态、不是自动执行器，也不能由旧“下一步”自动启动实验。

## 范围与使用边界

本仓库是**评审快照**，不附模型权重、原始PT/NPY/相机/视频/KV、编译.so、profiler二进制或连接凭据；不是拿到仓库即可原地运行的完整训练环境。部分源码保留原服务器路径和外部依赖。完整原始数值核验是在服务器上完成的，仓库凭证不能替代重新读取原始张量。

实际today50来源173项中，171个Python文件保留；两份编译库省略，对应C++/CUDA及vendor源与许可证已另附。Markdown本地链接已转换，指向未发布原件的链接会到[遗漏说明](review/OMISSIONS.md)，不伪造文件或PASS。具体变动、原始/导出摘要和遗漏均见manifest。

本地原tile代码/HW、原FAILED、冻结实验及`.pipeline/state.yaml/cycle.yaml/rules.yaml`未被此次导出修改；运行状态/账号级配置不作为公共仓库恢复配置。第三方来源与许可见[NOTICE](NOTICE.md)及各目录原LICENSE。
