# RoboTwin 六任务 × 每任务100次 × 三臂

## 范围和身份

用户已授权每任务100次这个目标。固定六任务：handover_block、lift_pot、open_microwave、place_object_basket、press_stapler、stack_blocks_two。T / 冻结ON3000连续投机 / 同权重PF2，每臂600局，合计1800局。结构、checkpoint、原RoboTwin tau、物理与命令执行保持，不追加训练。

按固定官方launch_client.sh的demo_clean、seed=0、test_num=100：每任务从10000起，依原expert_check过滤不可解场景，冻结前100条有效场景/seen instruction/完整初态；这100是有效评测attempt数，不是连续100个整数seed。T/C/P在同一GPU上用同一manifest；case次序固定，三臂轮换TCP/CPT/PTC，不以结果调整。

旧prefilter已有19条/任务可作为初态来源，但所有训练/CAL/DEV暴露身份必须保留。不能把整套称独立TEST；主表标扩大DEV，另报告未曝光部分的描述性分层。初态重用不等于新独立样本。旧DEV90完整结果和新增诊断不混为新1800个独立样本；扩大阶段按case/arm唯一键统计。

## 执行链

1. 先结束microwave两条原Target复现核查，比较所有命令及首次两policy，找出明确实施偏差则先修复，不以成功率低本身改成功判据。
2. 按官方expert逻辑补足每任务100条有效初态。独立manifest、GPU/renderer绑定、CPU/RNG/多相机证明；不生成teacher训练数据。冻结manifest后构建对应完整512-token BF16 text bank，T/C/P同等使用。旧bank不假定覆盖新增instruction。
3. 每个worker单模型，完整三臂case同卡；优先一GPU，另一GPU可在算子正式计时外接未开始的case。当前单worker显存约27GiB，A10040GiB不能直接再放一个同规模worker，不做同卡双模型超售。
4. 每10组case形成一个30episode shard。完整局自然成功或官方horizon；新attempt前预留整局上限与清理时间；资源/基础设施失败为PARTIAL，不自动重试或删掉。算法失败自然保留。
5. 每个shard终态立即汇总SR、配对胜负、成功长度分布/实际动作/各任务horizon罚分、全体与共同成功成本；C/P首次两S严格配对、路由和50有效更新核对。全部源/权重/normalizer身份固定。
6. 成本只作本轮插桩探索性数据。单层真实fixture/独占正式计时另外报告，不把两卡质量运行墙钟当同硬件无争用E2E结论。

## 有依据的时间估计与资源

刚完成90局GPU worker生命周期24199.408秒=6.722小时；按同吞吐线性估计1800局约134.44 GPU小时，尚不含扩大expert/bank准备。两卡各一个worker，理想约67.2小时墙钟；算子占卡、长失败和初始化会延长，预计约3–4天；单卡约6天。此为估计，不保证完成时间。

旧M40累计上限仅24GPU小时、硬截止10/22 01:26，不能覆盖这个新规模。本计划建议新增独立总上限160 GPU小时（包含准备/等待/失败/清理），最多10个日历日绝对窗口；具体T0和deadline在资源确认后登记，不重写旧ledger。每worker server/client各2 CPU线程；新增证据≤12GiB，少量代表fixture共享引用，不复制逐query全30层KV。

用户要求先诊断耗时而未批准160 GPU小时，因此先完成既有90局账本与一条完整成功任务profile，再修订吞吐估计/优化方案。未确认不自动启动1800局，也不租新GPU。当前诊断仍使用旧预算。

## 科学边界与日期

每任务100遵循官方评测规模，并不等于覆盖全部RoboTwin50任务或复现论文总体92.9%。微波炉原Target本身低SR必须如实与官方82% Easy的结果区分，不能将差距全部归因draft。

10/22只能用当时已有证据作阶段go/no-go；如果扩大尚未完成，标PARTIAL。10/23冻结结构/API/权重，后续只继续固定面板，不追加调参。10/28主表，11/01 18:00中文证据包；资源不足缩小结论，不保证赶上或篡改false。

来源：
- https://github.com/Robbyant/lingbot-va/blob/7c6ffa9bfc4b83582cafc860fab4c82cc7deeeeb/evaluation/robotwin/launch_client.sh
- https://arxiv.org/html/2601.21998v1
- ../dev90-20261021-r2/TERMINAL-SUMMARY.json
