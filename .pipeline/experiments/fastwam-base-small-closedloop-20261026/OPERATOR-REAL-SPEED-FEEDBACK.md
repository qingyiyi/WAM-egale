# 真实投机成本反馈给算子侧

2026-10-27 09:13新增用户要求与设备核对：prompt缓存已作为T/TF/PF共同优化独立在5090重测，见`../fastwam-full-triarm-20261027/TEXT-CACHE-v1/PLAN.md`，原累计7200秒预算内；当前投机会话占用独占测速窗口，实际释放读COORDINATION。没有改算子或VAE。沿加载链确认VAE已在GPU BF16，Base的`_encode_input_image_latents_tensor`直接调用eager `vae.model.encode`，动作compile未覆盖它；不能把约60ms误解释为CPU VAE。原共同成功3输入同步profile：TF VAE61.0ms（34.8%）、action54.0ms（30.7%）、text19.4ms、visual_prepare17.7ms、prefill12.9ms；action内部verify32.9ms、首尾target14.5ms、draft5.3ms。完整P50/P90与图见[耗时分解](../fastwam-full-triarm-20261027/LATENCY-BREAKDOWN-v1/REPORT.md)，阶段样本量与主速度分母不同。新缓存版会补每个共同成功组首观测的阶段profile。

2026-10-27 08:52统计口径更新：用户要求加速比仅计算三臂共同成功的原DEV来源组。当前Long/Spatial/Object/Goal完整action=1.24109/1.20897/1.21848/1.20718×，chunk=1.07165/1.05902/1.06363/1.06289×，源组N=16/19/20/20；[当前表](../fastwam-full-triarm-20261027/LIVE-RESULT.md)。这是CPU重汇总，无新GPU或数值路径变化。下方原始全输入形状、计数和profile仍作执行组织诊断保留，不能混作新共同成功集合的计数；新计数在各suite `SPEED-COMMON-SUCCESS-v1.json`。

本会话未改算子源码。真实TF/tau在5090：action1.232×，完整chunk1.069×；完整记录见REAL-SPEED-5090-RESULT.md。每chunk平均B7=3.05、B1=2.30，target状态行23.65；七行固定视觉KV、32动作token分别隔离，使用官方compile。

同步诊断一棵树verify约10.64–11.42ms，draft1.54–2.84ms，pack约.04ms，accept/commit约.31ms；这是一整动作网络的有限诊断，不是层级正式统计。VAE/text/视觉准备约占完整chunk主要固定成本。优先考察真实B7验证的整网收益与每round完整代价；单层收益不可乘投机倍数。现有persistent未胜compile，不默认接入。是否进一步做kernel由算子会话按独立授权决定。

60观测第一棵真实候选树、360接受边，B7对串行B1未见接受翻转；有限面板不代表严格等价，也不豁免新数值路径重新检查。


2026-10-27 Spatial补充：该suite独立checkpoint/tau=.01513551921，300对完整action1.19855×、policy chunk1.05632×；每chunk平均B7=3.05、B1=2.51667，共23.86667 verifier状态行。实际仍是LIBERO视觉KV98、私有动作32、文本129；Robo的三相机/14维路径尚未加载验证，不能共用这份实测形状收据。

主边界target约184.25ms，其中完整动作约61.34ms；投机动作约51.18ms。只优化动作循环的收益会被每次VAE、文本、视觉prefill等固定成本稀释。独立同步profile中三棵树verify约10.67–11.65ms、draft1.58–1.70ms、pack约0.042ms、accept/commit约0.32–0.35ms；仍是单个诊断调用，不能作为正式分阶段均值。更换kernel应重新实测完整action/chunk和实际接受路径，不将单层倍数与投机倍数相乘。

数值检查也有新边界：Spatial 60输入首树360条边中，B7/B1有1条阈值判断翻转、0条最终选择路径变化，原分数和tau已保留。Long既有20组canonical重复中，同硬件/物理/首个完整观测一致的7组，有2组首输出不逐位相同，其中1组最终胜负改变。根因尚未定位；这不是投机免除质量约束的理由。未来算子验收既要检查方向/更新，也要检查真实阈值附近的路径和完整闭环，不能只引用单层allclose。

来源：[Spatial速度审阅](../fastwam-full-triarm-20261027/libero_spatial/SPEED-REVIEW-v1.md)、[canonical重复复查](../fastwam-full-triarm-20261027/libero_10/CANONICAL-REPEAT-v2.md)。本会话没有修改或启用算子候选。

Long全量另一个成本信号：[在线位置分布](../fastwam-full-triarm-20261027/libero_10/ONLINE-ACCEPTANCE-POSITION-v1.md)显示step7的13315棵真实TF树均未接受两层，平均g=1.597，却仍验证7行；早期step1/4平均g约2.998/2.943。候选组织和末端策略可作为后续独立版本比较，当前冻结方法不改，不凭此表宣称未测收益。

2026-10-27 06:05：Object实际独占5090 action1.21848×/chunk1.06363×，固定trace TF1.06183×/PF1.03985×；step7示例仍只接受1层。tau=.01373201，905树深0/1/2=125/185/595，target B1=720/B7=905、rows7055。60首树360边B7/B1无接受翻转/路径变化，finite check非全局保证。未接入自定义算子；完整结果见 `../fastwam-full-triarm-20261027/libero_object/SPEED-REVIEW-v1.md`。

2026-10-27 CAL位置补充：Long/Spatial/Object在canonical校准状态的索引8均0通过；在线step7树同样无两层接受。不是只在草稿访问状态发生，见`../fastwam-full-triarm-20261027/CAL-LATE-POSITION-DIAGNOSTIC-v1.md`。方向预测相对直接沿用上一方向的离线CAL对照已保存，但尚无carry闭环或速度，不能据此声称新优化收益。本轮七行算法、tau及默认Torch保持不变。

2026-10-27 07:54：Goal真实5090 action1.20718×/chunk1.06289×、同长度TF1.06403×/PF1.04212×完成；B1=770/B7=905、rows7105（300chunks，G3000），首树360边0翻转/0路径变化。tau=.0145392660，完整动作T61.086ms/S50.602ms、完整chunkT176.144ms/S165.721ms。四suite投机测速均已退出，累计1870.287秒；没有接入自定义核。

2026-10-27共同prompt缓存四suite完成，见[最新表](../fastwam-full-triarm-20261027/CURRENT-RESULT-v5.md)。每臂75输入独立同步profile：TF文本.058ms/0.04%，VAE61.59ms/38.74%、动作55.84ms/35.12%、视觉准备18.09ms/11.38%、视觉prefill12.61ms/7.93%；动作内七行verify约57.74%，首步12.96%、尾部18.33%，draft8.79%、pack.22%。全部调用保持原Torch/compile，无算子集成，倍数不相乘。跨独立作业仍有路径差异，未定因；同进程cache开关225输入exact。Robo14维首四canonical已4/4，但14维实际B7调用形状仍待新适配检查，不能直接外推LIBERO尺寸。
