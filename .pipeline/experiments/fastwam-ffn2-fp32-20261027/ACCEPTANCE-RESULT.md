# 真实draft接受/十步传播诊断
OBSERVED：原已训练Long draft final3000，冻结tau=.015412352979183197。固定DEV观测列表前4项：task0/init47的chunk0/12/24和task0/init48的chunk0。它们只有一个任务、两个初态，不能算4个独立episode，更不能外推总体失败率。
以官方eager运行完整30层；仅B7 FFN2切换原严格DSO或连续FP32DSO。真实自然候选树及原selector保留。正式投机源码未改。
|固定观测|最终归一化动作relL2|maxabs|自身去噪路径改变|夹爪符号翻转|
|---|---:|---:|---|---:|
|0|0.337326%|0.0078125|False|0|
|1|0.346748%|0.0078125|False|0|
|2|0.414074%|0.0078125|False|0|
|3|0.386758%|0.0078125|True|0|

12棵固定真实树、72条边：1条阈值判定翻转、1个选择路径改变；4个完整十步动作中1个自身路径改变。所有路径仍合法提交10次scheduler更新。严格DSO720次局部FFN2字节一致，全部固定teacher和4次最终动作与官方eager一致；B1 canonical三路径4×10×3更新也完全相同。
具体：task0/init48/chunk0，去噪step7首边：eager=0.015974195674062，FP32=0.015297511592507，tau=0.015412352979183；选择depth0→1。该观察不能说质量改善或恶化。
去归一化后输出command最大绝对差按位置/旋转/夹爪分别为.00732422/.00233538/0（代码数值单位，不擅自当作米/弧度）。环境尚未执行，不能把它当实际末端位置偏移。
完整teacher方向relL2在.260767%–.318290%；依旧可能翻转阈值。全部观测/条件/有效视觉KV逐字节不变，CPU/CUDA/Python/NumPy RNG未被诊断额外改变；没有重调tau或改draft。
UNKNOWN：完整环境轨迹、成功率、更多任务/初态/其它suite影响；无独立TEST/非劣证明。
原始：remote/propagation-attempt01/results/INPUT-MANIFEST.json、FROZEN-QUERY-COMPARISON.json、CASES.json、query-*.pt及case-*-actions.pt。完整视觉KV未逐query复制，原观测+固定teacher来源可重建；不宣称存档包含每层完整KV。
