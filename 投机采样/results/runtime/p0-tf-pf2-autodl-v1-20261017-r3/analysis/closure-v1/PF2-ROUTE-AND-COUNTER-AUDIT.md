# 路由、计数与状态审计

OBSERVED：同一wrapper、同一冻结TF_ON、相同reset bank；C全S，P从首chunk S开始按S,S,T重复。reset/cache RPC不计周期，S全拒绝仍计S，只有完整policy成功返回才推进。所有完整policy实际提交50个合法scheduler更新。

十二局的COUNTER/AUDIT/FINISH_READY齐全，独立报告组件均PASS_COMPLETE_EPISODE_COMPONENTS，无证据缺口；六对物理/controller/实际Python、NumPy初始RNG、init state、model XML、双相机保存原件逐字节一致。首次分叉前两个S的RAM有效KV/RNG/张量比较每对REFERENCE_STORED/PASS_EXACT。

必须保留例外：reset/S0_before的六个lazy scheduler字段（video/action的sigmas、timesteps、training）只比较首次消费前的有效输入，原始全字段差异如实保存；native infer会无条件重建20/50时间表。S0/S1完成后比较全部捕获字段，不把例外扩展到其他状态。完整KV仅现场RAM比较，持久化摘要不是离线全KV重验；未保存的模拟器内部字段UNKNOWN。

S保留原O1/O2/O3、K2/B2、bonus、prefix_then_cost和50更新覆盖。T为canonical CFG2 action full50加原末尾工作，51次B2 action forward；video/cache/final工作分别记typed_forward_calls。没有删旧ObservedAPI断言或伪造O3事件。

OBSERVED：前两个S的6个描述单位中，风险/对照的接受深度接近。风险最大cost/tau分别.8473/.9372/.8967，对照.9543/.9907/.9955；同任务平均score差值方向不一致，不能据此建立“风险越大越贴门限”的解释。完整分析见../pre-treatment-score-result-v1/pre-treatment-score-analysis.json。

UNKNOWN：这不排除后续chunk局部敏感性，不证明现有score能预测任务风险，也不重新计算未保存的teacher velocity范数。不能反向利用六点拟合新阈值或按失败task硬编码回退。

RECOMMENDATION：保留原生产接受规则与四false；新诊断应明确区分teacher数值路径、接受draft和机器人闭环，不把PF2当bonus、环境回滚或质量证书。
