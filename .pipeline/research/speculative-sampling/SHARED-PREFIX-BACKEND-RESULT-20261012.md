# 共享历史KV数值差异：真实后端与存储对照结果

> 2026-10-12T18:37:58+08:00 | zh-CN | Asia/Shanghai | 投机采样研究线

共享KV数值差异已定位到attention计算路径变化：真实默认R为PyTorch Flash Attention；第0/8/29层、同一真实QKV下，S读取原CFG2共享池与复制CFG2历史逐位一致，相对R的单层RMS差异为0.06864%–0.10589%。原29输出哈希不变，数据、输出布局、局部cache版本/flags/RNG检查通过。S正式逐位门仍FAIL，尚无数值修复或性能验收；FP64抽样不支持把参考差异直接解释为数学精度下降。

本轮完成的是原因诊断。没有宣称共享kernel已经逐位对齐，也没有生成GPU速度、完整策略成功率或E2E结论。

## 现象与上下文

此前默认关闭的S使用CFG2原历史池，每路7个候选共用历史、当前Q/K/V仍是batch14；原R则把两路历史扩成14路后调用SDPA。旧S的29个最终输出均有限但0/29逐位通过，相对R的RMS差异为0.39543%–4.51987%。不能因attention入口设置为`torch`就认为原R调用MATH后端。

公共工作树原有HEAD为`92a633f`，model/server/config等已有修改在本轮之前已存在。本轮新增隔离实验入口和报告；原公共模型、原S kernel、历史失败、模型权重及`.pipeline/state.yaml`、`rules.yaml`与`cycle.yaml`的原状态保持不变。CPU审计验证了1589个既有文件和6个v2实现文件。

## 假设与验证

| 假设 | 结论与证据 |
|---|---|
| 新attention与原实际后端的浮点执行路径不同 | 已确认到执行路径层面。三层profiler均捕获`aten::_scaled_dot_product_flash_attention`；强制Flash与默认R逐位相同，原S和其他三个标准SDPA后端在同QKV下均产生不同输出。尚未唯一定位某一个舍入点。 |
| CFG行映射、有效slots顺序、历史K/V值有误 | 在本次三层首个真实candidate输入上排除。全部14行的历史K和V逐位等于对应CFG2池的有效slots，slots长度2144；当前Q/K/V直接使用原R实参。CPU负例也确认错误行归属及错误K值会被拒绝。 |
| 共用存储/索引读取本身改变数值或污染历史 | 在本次对照范围排除。同一个S kernel在原共享池及独立复制的两行紧凑历史上计算，三层共2,064,384个输出元素逐位一致、最大差值0；cache身份/版本和RNG不变。复制仍只有CFG2，不生成history14。 |
| 输出stride或BSHD/BHSD布局不同 | 实测输出layout原因排除。R和S都为BF16 BSHD `[14,16,24,128]`，连续，stride均为`[49152,3072,128,1]`。无需layout-only修复。 |
| 默认R随机性或观测包装扰动了原结果 | 本次检查排除。每层两次默认R重复逐位一致；强制诊断backend后flags恢复，三层局部cache版本与CPU/CUDA RNG不变；整个原capture的29个输出哈希与历史原R完全相同。 |

## 真实A100数值对照

固定原`T-init3-chunk14.pt`恢复和原A_off，先完成原接口准入再按每次模型调用安装观测包装。真实历史池为`[2,2160,24,128]`、有效2144，当前Q/K/V为`[14,16,24,128]`，CFG行映射为7个0再7个1。每层8次额外直接attention：2次默认重复、4种强制后端、共享原池S与复制CFG2历史S各1次。强制backend仅是诊断，正式参考仍为原默认R。

下表为`||output-R||₂ / ||R||₂`，绝非验收容差：

| 层（从0计） | 默认R重复 / 强制Flash | S共享原池 vs R | S复制CFG2历史 vs R | Efficient vs R | MATH vs R | cuDNN vs R |
|---|---|---|---|---|---|---|
| 0 | 逐位一致 | 0.07470% | 0.07470% | 0.07335% | 0.07804% | 0.07486% |
| 8 | 逐位一致 | 0.10589% | 0.10589% | 0.09434% | 0.13862% | 0.10601% |
| 29 | 逐位一致 | 0.06864% | 0.06864% | 0.06572% | 0.13313% | 0.06907% |


三层默认R实际backend均为Flash Attention。S对R有约9.3%–13.4%的输出元素字节不同（这不是RMS误差比例），最大绝对差0.0078125–0.015625；原共享池与复制池S完全同字节。Efficient、MATH、cuDNN也各自与Flash有小幅差异，说明相同attention数学定义和dtype不保证不同实现的BF16输出逐位相等。

S与cuDNN特别接近，但仍有228/331/376个元素不同，不能声称两者使用相同kernel或完全对齐。

## 为什么优化后出现差异

共享历史本身只改变取数位置，数学上仍是对“同一CFG历史+该候选自己的当前token”做同一个联合softmax。存储对照已经证明，本次真实输入上改为共享池取数没有附加数值差异。

当前优化同时把原PyTorch Flash Attention替换为自写Triton计算。原S代码/已存编译证据显示BN64分块在线softmax、FP32累加、概率转BF16再进行PV、exp2与融合运算；原Flash有自己的执行流程。浮点归约和中间舍入依赖操作顺序，改变分块/归一化/舍入路径即可能改变最后的BF16位。新kernel已经使用FP32累加，不能用“补FP32累加”作为未经验证的修复。

本轮已经隔离出不同执行路径的数值不兼容。具体是块宽/遍历、概率舍入、缩放融合、指数或它们的交互在主导，仍需单因素实验。此前逐层证据已看到差异沿模型传播；旧完整结果的最终误差确已达到约0.40%–4.52%，但本轮没有把每层/每轮放大量唯一归因。

## 参考差异与数学误差的区别

CPU独立用真实导出的BF16 Q/K/V，转换FP64后计算`softmax(QKᵀ/sqrt(128))V`。抽样3层×CFG两路（rows0/7）×heads0/11/23，共18个row/head切片，覆盖每个切片的全部16个query。这是高精度数学参照，不是GPU执行流程仿真：

| 层 | R相对FP64 | S相对FP64 | MATH相对FP64 |
|---|---|---|---|
| 0 | 0.17384% | 0.17372% | 0.17220% |
| 8 | 0.17189% | 0.17100% | 0.16531% |
| 29 | 0.17560% | 0.17488% | 0.16816% |


这些切片中S相对FP64没有比R更差；MATH更接近FP64却也不等于原Flash字节。因此目前应把问题描述为“与冻结原实现的数值兼容性未对齐”。抽样不能证明全部模型数学精度、更不能证明任务成功率或selector决策等价；正式逐位门保持不变。

## 实现与执行记录

v1包装安装过早，与已有“capture前必须原SDPA接口”检查冲突，GPU加载后、任何目标前向之前拒绝。已保存原v1源码、CPU日志和失败job；其写前计数确认0 target、0 direct，GPU含加载/退出耗时34.033726秒。不是原S精度失败的新证据。

v2改为在单次模型计算期间安装观测，保证capture前准入检查仍看到原SDPA；CPU新增早安装必须拒绝、正确时序和恢复回归，共10项通过。随后新独立job完成52次原target写前意图及52次主机返回、24次直接attention意图和24次返回，GPU结束前显式同步；闭合持久计数、原29个输出hash及资源清理通过。v2耗时78.666562秒，两次诊断共112.700288秒，低于原900秒上限。每个job仅启动一次GPU worker，失败job未复用。

CPU导出独立存储、字节hash、对照指标、18个FP64切片和冻结证据审计通过。实际导出199,372,797 bytes（约190.14MiB），包含三层真实prefix2/current14/输出及诊断输出，低于192MiB上限；无视图引用额外大storage。

0完整episode、0 EGL、0完整policy chunk、0训练。清理时GPU compute/graphics PID均为空、NVML实际分配显存为0，端口释放；即时utilization采样可能滞后，未将它误当作残留进程。未信号其他任务。

原累计预算继续携带：346/900 target意图，773.250910/2700 GPU秒，剩余554 target和1926.749090秒。旧完整计划要求585 target，无法直接塞进这个余额。本轮没有继续跑S性能或真实闭环。

## 修复方向与尚未通过的门

若继续追求逐位一致，以实际PyTorch Flash Attention计算实现为基准，尽量只改K/V访存地址映射为CFG2共享历史+各候选当前K/V，保持原归约、softmax与舍入顺序；先小规模真实输入门，再覆盖全部30层/29query、state/RNG、12类失效、13行隔离，最后测完整verifier速度。本轮未定位唯一主导舍入点；不通过改参考backend或放宽容差宣称修复。原累计余554target，小于旧完整585计划，后续需重新安排验证预算并保留全部门。

正式S状态仍是FAIL。S全30层/29query的新数值修复、全程state/RNG、12类失效、其他13行隔离、完整verifier速度和真实任务闭环均没有因本次观测而获得PASS。无需更改公共架构，研究记录将原backend/stride的UNKNOWN更新为本次实测结果。

## 可复查证据

- [成功观测及资源清理](../../../投机采样/results/runtime/p0-shared-prefix-observe-v2-20261012-182534/FINAL_RESULT.json)；[计划与源码冻结](../../../投机采样/results/runtime/p0-shared-prefix-observe-v2-20261012-182534/plan.json)；[CPU10项关](../../../投机采样/results/runtime/p0-shared-prefix-observe-v2-20261012-182534/preflight/CPU-GATE.json)。
- [第0层原始metadata](../../../投机采样/results/runtime/p0-shared-prefix-observe-v2-20261012-182534/observation/layer00-metadata.json)；[第8层](../../../投机采样/results/runtime/p0-shared-prefix-observe-v2-20261012-182534/observation/layer08-metadata.json)；[第29层](../../../投机采样/results/runtime/p0-shared-prefix-observe-v2-20261012-182534/observation/layer29-metadata.json)；对应`layerNN-tensors.pt`保留全部输入与输出。
- [CPU导出与FP64独立复算](../../../review/OMISSIONS.md)；[审计代码](../../../review/OMISSIONS.md)。
- [原S正式数值失败](SHARED-PREFIX-CFG2-NUMERICAL-RESULT-20261012.md)；[此前CPU方案](SHARED-PREFIX-NUMERICAL-PLAN-CPU-20261012.md)。
- [本次v1前向前拒绝](../../../投机采样/results/runtime/p0-shared-prefix-observe-v1-20261012-181929/FAILED.json)；[v2安装时序](../../../review/OMISSIONS.md)；[CPU回归](../../../review/OMISSIONS.md)。
