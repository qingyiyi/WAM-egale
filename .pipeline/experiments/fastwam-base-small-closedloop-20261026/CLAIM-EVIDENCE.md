# FastWAM 投机迁移：主张与证据边界

核对日期：2026-10-26。仅维护证据及受影响段落，不生成论文正文。

当前速度口径（2026-10-27用户补充）：只计算原测速来源中target/TF/TF+PF都成功的配对组，失败不参与加速比；质量仍用全体有效集合。下列历史速度数字保留追溯，当前主统计以[共同成功汇总](../fastwam-full-triarm-20261027/LIVE-RESULT.md)和各suite `SPEED-COMMON-SUCCESS-v1`为准；不得把条件性结果写成全体任务速度收益。

2026-10-27四套原版全量闭合：[6000局完整审阅](../fastwam-full-triarm-20261027/FOUR-SUITES-RESULT-v1.md)，T1950/2000、TF1945/2000、PF1944/2000，Goal新完成493/490/491（各500）。新增失败与成功长尾保留，原面板暴露/数值重复性边界不变。用户另要求的共同prompt缓存正在独立版本检查/测速；旧全量质量不能改称新缓存版本质量，缓存属于共同工程优化，不是方法新增贡献。

|近邻|论文已包含的机制|本轮可区分的对象|当前证据状态|
|---|---|---|---|
|[Fast-WAM v2](https://arxiv.org/html/2603.16666v2)|训练保留视频建模；部署只对当前观测建立视觉条件，不生成未来视频|固定十步原模型上的候选状态、验收、合法scheduler提交|固定源码接口已核对；官方缓存与compile均属于基线，不算本方法贡献|
|[1Step-WAM v1 §4.1/4.3](https://arxiv.org/html/2610.15625v1)|冻结backbone，用轻量校准头改初始速度以逼近多步teacher终点；另有逐层视觉/动作流水|本轮拟合指定状态/时间的瞬时target方向，在线验证候选去噪更新，双方仍走同一个十步scheduler|仅“冻结target＋小模块”或视觉/动作流水不足以主张创新；本轮未复现其模型/结果|
|[Realtime-VLA FLASH v1 §3.3–3.5](https://arxiv.org/html/2605.13778v1)|draft生成动作chunk，重用先前完整路径KV；并行验证多个时间点的一致性，选取可执行动作前缀并按阶段fallback|本轮验证的是七个候选去噪状态，提交scheduler更新前缀；每轮当前观测条件固定，无跨chunk旧KV复用|不能只声称“连续动作并行验证”；本轮未复现FLASH，不将其最短fast-path时间当完整baseline|

本轮待验证：轻量方向树能否在新CAL下形成足够的接受深度；七行实际验证、打包、同步和首尾成本能否小于逐步target；在完整失败保留的闭环面板上，成功率及共同成功自然trace是否仍可接受。历史Long接受重放只能提供条件速度证据。

本轮轻量预测器显式输入为动作状态、上一target/已提交方向和时间；观测/文本/proprio只通过方向隐式影响draft，完整条件供target使用。不是增加视觉编码器。CAL分位点不是在线接受率或成功率保证，连续近似接受不具备已证明的严格分布保持性质。

受影响段落清单：研究动机的平台选择；与少步蒸馏/FLASH的相关工作边界；方法中的预测对象与提交单位；实现中的Base接口、B7隔离与成本；实验的固定身份、数据隔离、质量/自然trace/完整成本。待实测后再决定能写的结论。算子侧证据由另一会话独立补充，普通batch/cache/官方compile不计新增贡献。

2026-10-27补充：[论文v2表1/2](https://arxiv.org/html/2603.16666v2)的原Base比较目标为LIBERO Spatial/Object/Goal/Long=98.2/100.0/97.0/95.2%，Robo clean/randomized=91.88/91.78%。[结构化身份](PAPER-BASE-REFERENCE.json)。README中Optional-IDM、shift1的first_frame 95.8% Long属于另一权重，不混作本轮原Base目标。当前完整canonical仍在评估；首轮DEV不能称论文成功率复现。

2026-10-27全量Long证据：固定十步Base，T477/500、TF477/500、PF473/500；TF完整平均命令265.66对T265.976，TF共同成功468对的P90长度比1.021。TF仍有9新增失败与9救回，且4个共同成功长度比>1.3（最差1.749）；PF5个（最差2.221）。不能据均值接近宣称逐局trace或无损保持。独占5090固定DEV输入action1.232×/chunk1.069×，自然长度回放仅固定4组，不是完整500组或仿真E2E。参见[全量结果](../fastwam-full-triarm-20261027/libero_10/RESULT-REVIEW-v1.md)和[长尾](../fastwam-full-triarm-20261027/libero_10/TAIL-REVIEW-v1.md)；四权限仍false。

2026-10-27数值重复性边界：Long既有DEV/full target重复观察到了同硬件、同初态及首观测下的小幅首动作差异和一例胜负变化，[CPU复查](../fastwam-full-triarm-20261027/libero_10/CANONICAL-REPEAT-v2.md)。尚无定位根因或独立重复对照，不能把所有配对胜负差异当作算法因果效应，也不因此删除投机新增失败。正式质量段落需显式交代重复性与统计不确定性。

2026-10-27后续全量：Spatial T481/TF481/PF482（每臂500），target低于Base论文2pp；Object T499/TF497/PF498，target低.2pp。Object各500组初始图像全部逐位一致，TF仍新增2败且2个共同成功trace>1.3倍；不能写成均值接近即逐局不变。5090 TF action/chunk分别Spatial1.199×/1.056×、Object1.218×/1.064×，均沿原十步/官方compile；固定长度policy replay和真实仿真E2E继续分列。详见[Spatial](../fastwam-full-triarm-20261027/libero_spatial/RESULT-REVIEW-v1.md)、[Object](../fastwam-full-triarm-20261027/libero_object/RESULT-REVIEW-v2.md)。Goal/Robo尚无全量结论，四权限仍false。

2026-10-27 08:45 方法增量边界：[四套CAL对照](../fastwam-full-triarm-20261027/CAL-CARRY-COMPARISON-v2.md)显示TF相对直接沿用上一target方向，早期索引1方向分数明显降低，但索引4几乎相同，索引7当前阈值通过率反而较低，索引8双方均0通过。这只使用保存的canonical状态与标签，未运行carry闭环。轻量网络是否比零训练方向外推增加在线收益仍需独立对照；当前真实TF质量与速度成立于已测方法，不据离线表声称carry更好或等价。现有checkpoint、tau、七行树和运行作业不变。

2026-10-27用户扩展同一篇论文的研究范围：面向flow-based robotic action models，后续有时间和资源时补StarVLA-PI、StarVLA-GR00T跨模型验证。两者尚未执行，代码/权重与动作接口待核对；当前FastWAM证据不能外推为跨模型有效性。新增受影响段落：研究范围、方法适用条件、跨模型实验与局限性。见[长期决定](../../memory/global/requirements/decision-研究范围扩展至基于流的动作模型并候补StarVLA跨模型验证.md)；不自动重写正文。
