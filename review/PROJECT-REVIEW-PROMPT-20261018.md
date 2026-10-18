# WAM 最新整体进度与下一步：网页审查 Prompt

请用中文审查 `qingyiyi/WAM-egale`。该仓库只维护main作为当前本地进度镜像：若用户给出固定commit，严格使用它；否则先读取main并记录当前完整commit，本次审查固定该版本，不在中途切换。你的任务是读证据、核验相关工作并给服务器 Codex 一条有限可执行的路径；本 Prompt 不授权你启动 GPU、训练、采集、TEST、删除记录或扩大预算。

## 先读当前入口

1. `review/CURRENT-PROGRESS-20261018.md`。
2. `.pipeline/cycles/wam-research-20261016/STATUS-20261018.md`：整体进度、时间口径、当前任务与缺口。
3. `.pipeline/experiments/long-morning-20261018/FINAL-SUMMARY.md`：Long首轮完整500组三臂。`review/RESULTS.json`仍是旧paused72历史科学快照，禁止当成当前覆盖/资源状态，也不要改写它。
4. `.pipeline/experiments/operator-torch-alignment-20261018/RESULT.md`与`performance/RESULT.md`、全部性能CSV、`performance/CPU-INPUT-IDENTITY-AUDIT.json`。数值阶段报告有明确当时状态，当前性能看performance子目录，不误读成“仍未测”。
5. `.pipeline/experiments/libero-four-suite-20261017/CURRENT-RUNNING-SUMMARY-20261018.json`和当前Cycle四文；远端快照有时效，不根据旧数断言仍实时。
6. `review/OPERATOR_REVIEW_PROMPT-20261018.md`、`review/UPDATE-OMISSIONS-20261018.md`。

需要解释源码时继续读：

- `投机采样/experiments/operator_torch_alignment_v2/{setup.py,integration.py,worker_full.py,worker_performance.py,worker_full30_performance.py,csrc/standalone_flash_sdpa.cu}`。
- `投机采样/wam-p0-worktree/wan_va/modules/model.py`、`wan_va/wan_va_server.py`。
- `投机采样/experiments/wam_candidate_tile_v1/`、`wam_candidate_tile_guarded_v2/`、`wam_static_prompt_baseline_v2/`。
- 旧质量契约与接受器按CODE-MAP及真实import链追踪，不依据目录名猜测。

先列实际成功读取的文件；缺文件、缺原始tensor、无法打开GitHub时明确UNKNOWN，不声称完成独立重放。公开仓库包含源码与结构化结果，不含权重、PT/相机/完整KV/.so。

## 必须保持的判断边界

- TF64结构/API继续固定；当前权重和训练配方没有旧质量证书。Long首轮连续TF的96.8%与Target96.6%不证明TF优于Target/MLP或非劣；PF2为94.6%，不能因其他小面板局部改善就设默认。
- Long有少数成功长尾、训练/DEV暴露和首两S严格配对FAILED；成功率、全任务800罚分、共同成功长度分别报告。不得剔除长尾后用子集速度替代全体结果。
- 两端模型/算子对齐是不同契约：新独立算子相对同形状Torch的有限逐位一致，不等于原B2与B14整个teacher模型逐位一致，也不等于新机器人质量通过。
- 新v2仅去掉fast_math，真实108attention/54block/六full30/三完整canonical policy配对有限逐位通过。attention速度接近Torch，但9/54 block、2/6 full30慢超过5%，三样本且资源异常保留，不能说所有形状稳定一致。
- 原计时期间本机CPU被gem5/QEMU严重挤占；随后模拟器已退出。未在CPU减负后的新独占窗口复测，不能事后把所有慢点归因CPU，也不能把旧时间更正掉。
- 本机继续算子工作，新机既定Goal/Spatial满载、Object等自然槽；用户取消进一步开发迁移。不要再规划整体迁移或建议结束他人GPU任务。reset优化已经全臂启用。
- B14是七候选各Q16、CFG2，不能当成跨候选Q112序列；private KV不共享。Linear扩行现成库已经部分做到，普通缓存/Graph/shared-prefix均需作为共同强baseline。
- warm自然episode探索性时间不等于正式独占/cold E2E；target calls和单边接受率不等于加速。主机制成本SUM(T_round)/SUM(g)，并列完整chunk/任务成本与质量。
- engineering_quality_screen、independent_test_complete、noninferiority_established、formal_performance_permission四项均false。旧FAILED/阈值/retention、paused72身份、原预算和原tile保护不改，不能为了PASS调规则。

## 请回答的五件事

1. 给claim—当前证据—缺口—允许表述矩阵，区分OBSERVED / HYPOTHESIS / UNKNOWN / RECOMMENDATION。哪些可以写中文方法，哪些只能写探索性结果？
2. 下一步优先完成什么，为什么？请在“Torch替换基线的block慢点/主机干扰定位”与“开始候选调度原型”之间给一条默认路径。列最小真实输入、强baseline、CPU/GPU/墙钟/存储上限、验收与停止线；不要先建议大型30层融合、重编整个Torch或大网格搜索。
3. 如何验证慢点来自CPU提交/launch/allocator/时钟或非attention算子？新实验只能前瞻登记，保留旧全部慢点；如何避免把3个样本的偶然快点写成创新？
4. 质量线在四套既定评测完成前后分别做什么？连续TF/PF2现有信号如何解释，哪些审计/长尾/必要少步Target对照仍需补？不自动建议架构sweep、无限训练、PF3/4、再跑到通过或扩大环境。
5. 真正MegaTransformer增量需要怎样证明超过最佳B14 shared-prefix/O3/Graph和共同cache？明确候选调度与投影cache的两个轴、数值/状态门、有效提交成本和完整任务质量关系。普通batching或cache收益不冒充新贡献。

若参考文献，只查少量原论文、固定官方源码和NVIDIA文档；标明版本、实现重合与可迁移到A100/SM80的具体条件，不默认TMA/WGMMA可用。需要CVPR注册/摘要/正文日期时重新查2027官方页面，不套往年。

最后给一个24–48小时默认计划及到10/22 go-no-go、10/23冻结、10/28主表、11/01 18:00北京时间中文证据包的关键路径。区分已授权工作和待申请作业，最多列3项真正缺失的信息。不保证通过或录用，不让宽泛提纲替代可复核的下一步。
