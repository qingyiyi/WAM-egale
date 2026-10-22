# 单层Transformer：一个persistent计算kernel

当前入口：[whole_layer.py](whole_layer.py) → [whole_layer.cu](csrc/whole_layer.cu)。Python绑定27组原参数和KV，C++准备workspace，一次cooperative计算launch完成一层；workspace初始化与host开销另计。没有设备端子kernel，不是30层大融合。

## 源码阅读顺序

1. `whole_layer.py`绑定参数，普通调用返回hidden；`debug=True`提供外部核验所需中间量。
2. `whole_layer.cu::whole_layer()`按模型顺序调用所有阶段。
3. `action_projection/context_linear/action_ffn1_wide/action_ffn2_parallel`及[cutlass_gemm.cuh](csrc/cutlass_gemm.cuh)解释GEMM tile与原BF16边界。
4. `norm/point`解释行归约与向量化逐元素计算；[fa2_stage.cuh](csrc/fa2_stage.cuh)解释内部FA2、合法共享历史和私有KV。
5. `begin_stage/claim_tile/finish_tile`只服务仍动态分派的路径；静态阶段不再重复初始化队列。

## 输入、输出与计算链

hidden为BF16 `[B,Q,3072]`，context为BF16 `[B,512,3072]`，temb沿原BF16/FP32调制契约，RoPE保持原算术。输出与hidden同形状、同dtype。B14是14个独立Q16实例；线性层可展平为M224，attention private可见性仍逐候选限定。

`Norm1/调制 → self QKV/RMS/RoPE/FA2 → 投影/残差 → Norm2 → cross QKV/RMS/FA2 → 投影/残差 → Norm3/调制 → FFN1/GELU/FFN2 → 门控残差`。

|B14阶段|实际GEMM M/N/K或工作量|tile及任务|warp职责|
|---|---|---|---|
|self Q/K/V、self out、cross Q/out|224/3072/3072|M64×N128，96输出tile；两个K1536依次计算|3–6做CUTLASS MMA，8warp共同写回|
|cross K/V|7168/3072/3072|M128×N128、K64三级流水，各1344tile|3–6做MMA，8warp写回|
|FFN1|224/14336/3072|M128×N256、K32三级流水，112tile；FP32结果分半复用shared写回|8warp均做MMA和写回|
|FFN2|224/3072/14336|48个M128×N128输出tile×两K7168段＝96计算任务|3–6做MMA，按原顺序提交两段BF16|
|self/cross attention|每实例Q16、H24、HD128|每CTA两个独立4warp组；self为受guard的G1/G2，cross仍独立实例|两个组各有workspace/参数及barrier3/4|
|Norm/RMS|每行3072|普通Norm一行；RMS两行/CTA，静态分配|原128线程归约树；两4warp组各算一行|
|pointwise|2048逻辑单位/tile|静态block-stride；调制/残差/GELU每次4元素；RoPE每次2复数对|8warp参与，RoPE保留FP64算术|

B2 action仍保留原小M32、M64×N64/K32兼容路径；本轮B2/video属于精度覆盖，没有用B14速度外推其性能。

## shared、register与同步

当前108个常驻CTA，每CTA256线程/8warp；不是每个阶段必须凑108个任务，也没有跨阶段ready队列。不同阶段按自己的tile工作量执行。

GEMM使用CUTLASS设备主循环（SM80 cp.async/MMA），不是在kernel内调用cuBLAS。通用写回用136列FP32 shared stride与向量写回；FFN1两半结果复用同一写回缓冲，避免为N256输出扩大整层shared预留。中间activation仍按原BF16/FP32边界写global workspace，一个kernel不等于全部数据留在shared。

FFN2先并行算两个K7168段，第0段写BF16；经grid同步，第1段按`(FP32+bias)+第0段BF16`归并后再写BF16。保持原舍入，不能用一次FP32总和替换。

静态point/norm/FFN去掉不再使用的入口队列初始化及重复全局屏障，复用上一阶段出口的输入就绪保证；双组FA2只用CTA同步发布本CTA描述符。所有必要的阶段出口、FFN2分段归并和动态队列同步保留。B14空cache-insert阶段直接返回，候选private KV从不写进公共历史；B2真实cache更新不变。这些改动不是跨阶段DAG。

G1/G2只在实际共享条件/历史的候选间复用prefix，每行private suffix仍有独立可见性。非16对齐history回退G1；当G2会增加K tile或实际CFG引用不匹配时也回退。此前short/drop1曾有2个attention中间元素差1.49e-8（最终层输出相同），原严格失败保留，不能将其忽略为PASS。

## 当前验证与剩余差距

[夜间证据入口](../../../.pipeline/experiments/single-layer-async-gemm-20261020/overnight-alignment-20261022/PLAN.md)、[同轮完整层计时](../../../.pipeline/experiments/single-layer-async-gemm-20261020/overnight-alignment-20261022/WHOLE-LAYER-COMPARISON.csv)、[最终身份](../../../.pipeline/experiments/single-layer-async-gemm-20261020/overnight-alignment-20261022/SELECTED.json)。最终选择静态入口清理版，RMS两warp/行、小投影K并行等退化候选均已撤回，失败/负结果在证据目录保留。

9真实签名（B2action/B14action/B2video×3KV）及21结构变体有限逐位通过，包括23共有中间量、普通/debug、有效KV、输入/RNG与B2合法cache状态转移；不是所有输入、所有层或线上完整模型证明。fixture是保存后的逻辑tensor契约，不能声称恢复全部原GPU alias/stride。

远端独占A100-SXM4，两个独立进程、固定warmup、交错完整层Graph：最终B14短/中/长约2.296/2.453/2.476ms，较夜间起始版本耗时降低8.5%/9.7%/16.8%。相同uncached工作量的guarded强对照约2.070/2.187/2.201ms，最终版仍慢10.9%–12.6%；不是对最强路径已对齐，也不是30层/任务E2E。

[资源控制](../../../.pipeline/experiments/single-layer-async-gemm-20261020/overnight-alignment-20261022/resource-envelope-control/RESULT.md)证明固定资源限制部分原语：更多可驻留CTA使独立CrossK RMS约116–119→94us，但原版约70us；不能据此把更多CTA直接搬进当前整层。当前静态资源108CTA/1CTA每SM、100096B shared、255寄存器/线程、localSize936B；localSize不是动态spill或HBM流量。

晨间[NCU与负结果](../../../.pipeline/experiments/single-layer-async-gemm-20261020/morning-alignment-20261022/RESULT.md)已补齐：当前local路径232.337MiB；两项源码对照无收益，全局LTO慢约16%，均撤回。LTO让FFN变快但Cross-K/V嵌入整层明显变慢，下一步应限制各阶段编译影响，而非全局打开LTO。后续DAG需区分缓存命中，缓存命中时context K/V投影不是可提前领取的任务。当前未实现ready-DAG，不把普通shared-prefix/G2或缓存本身写成新贡献。baseline、投机生产和原tile保持只读；全局质量及正式TF性能四项权限仍false。

[简要带读Prompt](READ_CODE_PROMPT.md)。
