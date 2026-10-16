# Debug-SHARED-PREFIX-BACKEND：共享KV输出与原SDPA不一致

> 2026-10-12T18:37:58+08:00 | zh-CN | Asia/Shanghai

共享KV数值差异已定位到attention计算路径变化：真实默认R为PyTorch Flash Attention；第0/8/29层、同一真实QKV下，S读取原CFG2共享池与复制CFG2历史逐位一致，相对R的单层RMS差异为0.06864%–0.10589%。原29输出哈希不变，数据、输出布局、局部cache版本/flags/RNG检查通过。S正式逐位门仍FAIL，尚无数值修复或性能验收；FP64抽样不支持把参考差异直接解释为数学精度下降。

## 症状与上下文

原S完整29query逐位门FAIL；当前任务要求精度对齐或给出原因。公共工作树原有HEAD `92a633f`、原model/server/config的已有修改保持不变。本轮只增加隔离实验和研究证据；legacy pipeline保护状态不改。

## 假设、根因与验证

真实默认backend已确认Flash Attention。固定同QKV、保持原S数学kernel时，将共享CFG2池换为独立复制CFG2历史，三层2,064,384个元素逐位相同；换为其他标准SDPA backend则与原Flash不相同。全部14行历史值、原输出布局、局部cache/RNG/flags、默认重复和整个原29输出hash检查通过。

已确认根因位于更换attention浮点计算实现带来的数值兼容性差异。具体唯一主导舍入点尚未确定；不把FP32累加缺失、共享存储污染或布局错误作为本次实测根因。FP64真实输入抽样中S相对数学参照没有比R更差，参考字节差异不直接等于数学精度或任务质量下降。

## 修复建议与状态

若继续追求逐位一致，以实际PyTorch Flash Attention计算实现为基准，尽量只改K/V访存地址映射为CFG2共享历史+各候选当前K/V，保持原归约、softmax与舍入顺序；先小规模真实输入门，再覆盖全部30层/29query、state/RNG、12类失效、13行隔离，最后测完整verifier速度。本轮未定位唯一主导舍入点；不通过改参考backend或放宽容差宣称修复。原累计余554target，小于旧完整585计划，后续需重新安排验证预算并保留全部门。

新增观测包装第一次安装时序有误，原接口门在0 target/0 direct时拒绝；失败保存。v2把包装限定在单次计算内，新增对应CPU回归，10项通过；有界GPU诊断52 target/24 direct通过，闭合计数和清理通过。两次诊断总GPU112.700288秒，不覆盖旧失败。当前结论为诊断完成，S逐位修复/性能验收未完成。

## 架构影响与完整证据

无需改公共架构，更新研究记录中的backend/stride认知。详细假设表、所有数值、限制、预算和原始证据见[完整报告](../research/speculative-sampling/SHARED-PREFIX-BACKEND-RESULT-20261012.md)；独立复查见[CPU审计](../../review/OMISSIONS.md)。
