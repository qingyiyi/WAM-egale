# Tensor Core与RoPE源码核验

## 已观察：当前活动二进制

本次核对修改前的已验证活动DSO `41af9a655a2387e4871eb5ab5dee1a9ac0cf613fd35c66f1f076fc5ef0f20b41`，不是仅看待编译源码。`cuobjdump --dump-sass`显示action_projection、context_linear、FFN1/2、attention_dual和FA2兼容入口均包含`HMMA.16816.F32.BF16`。详见[原始静态审计](ACTIVE-TENSOR-CORE-AUDIT.json)。指令出现数为静态统计，不是运行次数、吞吐或Tensor Core利用率。

源码链：`optimized/csrc/cutlass_gemm.cuh`的`OpClassTensorOp/Sm80`；`fa2_stage.cuh:239–263`的B14 cross direct与self grouped两个调用；生成`fa2_device_generated.cuh`的whole_grouped_dual中QK调用2359/2427、PV调用2407/2469，whole_direct_dual中QK2805/2872、PV2852/2913；均使用Kernel_traits::TiledMma。固定vendor kernel_traits.h:30–34选择`SM80_16x8x16_F32BF16BF16F32_TN`。生成行号对应上一轮modulation build，不随当前源改写历史。

结论：当前投影及attention两次矩阵乘加使用Tensor Core。Softmax、mask、归约、RMS、RoPE、加载/写回不因此都成为Tensor Core运算。WAM当前仍存在整层性能差距，这次静态核验不能将原因归为利用率低或证明利用率高。

## RoPE实际公式

当前每head128维分为三轴44/42/42维，对应f/h/w轴22/21/21对。对轴a的坐标p_a，频率omega[a,i]=10000^(-2i/d_a)，角度theta[a,i]=p_a*omega[a,i]。代码先用double频率乘grid_ids，再转换float角度并生成complex64的cos+i*sin，原model.py:243–286。

对RMS后每对BF16元素(x0,x1)，旋转为y0=x0*cos(theta)-x1*sin(theta)，y1=x0*sin(theta)+x1*cos(theta)。当前原模型将元素转double进行复数乘法，最终转回原dtype；自有算子保留已验证的FP64运算及FP32/BF16转换边界。旋转作用在每个self-attention Q/K的相邻元素对，V和cross-attention不做RoPE。

在理想实数表达中，(R(m)q)^T(R(n)k)=q^T R(n-m)k，因此Q/K点积包含相对位置；数值实现存在原舍入，不能据此声称浮点严格正交。三轴分别应用对应坐标。RoPE位置来自grid_ids，不是本轮扩散sigma的调制参数。

这些是源码解释；本轮RMS+RoPE候选的正确性和性能另见实际RESULT，不混为已完成优化。
