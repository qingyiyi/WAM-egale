# 原精度失败定位：复用已保存对照，0新GPU调用

OBSERVED：主算术源df8b386…，9个真实block签名来自既有捕获；共享输入/条件的多个历史长度不是独立统计样本。这里读取validate-1原JSON，没有重新运行teacher、完整模型或机器人。

|输入|首个已保存DAG差异|同输入Q不同元素|同输入FFN2不同元素|cross K/V位相等|
|---|---|---:|---:|---|
|short-B2-action|self_q_norm|68/98304|39076/98304|True/True|
|middle-B2-action|self_q_norm|68/98304|39012/98304|True/True|
|long-B2-action|self_q_norm|68/98304|39091/98304|True/True|
|short-B14-action|self_q_norm|141176/688128|125565/688128|True/True|
|middle-B14-action|self_q_norm|132528/688128|121406/688128|True/True|
|long-B14-action|self_q_norm|135744/688128|123031/688128|True/True|
|short-B2-video|self_attention|0/786432|236110/786432|True/True|
|middle-B2-video|self_attention|0/786432|237080/786432|True/True|
|long-B2-video|self_attention|0/786432|236793/786432|True/True|

OBSERVED：相同输入控制直接将新原型的modulated1交给原to_q、将新gelu交给原FFN2，再比较对应新输出，因此这些差异不能全部归咎于上游误差传播。B2 action的Q只有68个元素不同，B14的Q约13.3万–14.1万个；FFN2九组均不同。它们的hidden误差不能解释成物理动作容差。

OBSERVED：三份video输入的所有已保存self-attention上游量（包含Q/K norm、V、RoPE）逐位相同，首个已保存差异在self attention。需要同时处理attention边界与GEMM契约，修好一个FFN2不足以让整层通过。九组cross K/V及其原条件输入对照通过；不能笼统宣称所有GEMM都不可能匹配。

UNKNOWN：主审计没有保存每个原投影pre-norm输出，故“first_saved_DAG_difference”不是已经证明的全图最早根因。独立Q控制给出更早GEMM算术差异证据，但不会追溯补造缺失主路径tensor。video attention边界仍需区分原Flash归约/在线softmax顺序、有效KV顺序和可见性，不能单凭小误差指定某个浮点指令为根因。

本机公共cuBLASLt metadata重建显示B2 Q splitK1、B2 FFN2 splitK7、B14 Q/FFN2 splitK2；这不是远端旧调用现场拦截。B2 Q在splitK1也有差异，所以“加splitK即可全部解决”没有依据。原四种累加公式已失败，不能继续无限猜公式。

RECOMMENDATION：原baseline保持冻结可用。后续设备原语先按相同输入分别处理小M Q、FFN2、Flash attention三个数值边界，再接回整层；保留原BF16舍入、FP32打分和全部KV/RNG要求。若在固定Torch库契约下无法一致，先登记新数值与闭环验证路线，不能把本轮match-v1或普通allclose当成原质量证书。
