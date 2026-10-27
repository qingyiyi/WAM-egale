# FastWAM小规模迁移接口 v1

固定源7faa711、Base8eaceeb；旧WAM参考89ba588。实际历史导入：tf_effect_train_v1/adapter → target_trace_draft_train_v1/adapter.ActionPredictor → draft_architecture_v1/models.PositionTransformer；loss来自target_trace_draft_finetune_v1/train.loss_value；首轮selector保留bonus_acceptance_v2 depth_then_cost，PF2不加入。旧checkpoint、normalizer、tau均不搬用。

可复用：单层width64/heads4/FFN256的双头残差方向预测、episode分组采集、七状态树/FP32步长加权L2、最长合法前缀再比总代价、fallback与leaf bonus。新实现32个时间顺序动作token，输入state224+previous224+scalars3=451，输出2×32×7，参数54,094。只有动作状态、上一方向、时间显式进入draft；文本、观测、proprio经真实target方向隐式影响输入，与旧轻量设计一致；保存完整条件以便重建teacher。不是新增条件encoder或架构搜索。

必须适配：原B14/CFG改B7单路，7候选各32token独立attention；30层hidden1024、attention3072、FFN4096；128 T5+1 proprio；本次观测固定逐层视觉KV（LIBERO98位置），动作KV每次重新计算。十步BF16 timestep与delta来自官方scheduler，不由timestep差倒推delta。seed42每次CPU重建噪声。无未来视频生成或末尾t=0持久提交。proprio8不追加动作token。

保持官方camera180度旋转、两相机各224再横拼448，官方proprio/动作统计量，gripper反归一化→2x-1→取反→sign各一次。Long env256渲染、静置30、完整700动作步、32预测/10执行/重规划；原MSAA保留，不复用旧无MSAA改版。初态物理/RNG与像素签名分别记录，像素非逐位等不能暗称严格图像配对。

canonical运行官方infer_action；collector只替换等价十步循环且先对原eager完整动作逐位核对，再检查官方compile路径。新增文本缓存0；双方同样每次编码文本与VAE/视觉prefill。训练label为访问状态真实target方向，不是数据集动作。FIT每步存state/previous/target/t/d和has_previous，训练排除首步；canonical数据与未采集的draft/在线分布明确区分。

暂不迁移：persistent/kernel、旧full512跨chunk bank、MoE、PF2、减少verifier行、跨轮根方向复用、蒸馏替换target。bootstrap一次target全部计费，末尾不足三步按canonical走，bonus只执行合法下一更新，总g=10。

CPU oracle检查通过仅说明控制流；GPU还须实际B1/B7方向/更新、候选隔离、条件只读和compile一致性。容差沿先前FastWAM实测契约(relative L2≤1%，maxabs≤.05+.02·参考峰值，scheduler BF16两舍入ULP包络)，逐项记录实测值，不在失败后扩大。
