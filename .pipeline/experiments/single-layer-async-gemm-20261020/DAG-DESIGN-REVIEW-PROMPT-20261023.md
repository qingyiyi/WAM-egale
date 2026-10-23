# 给网页版的项目现状与用户想法：请独立判断下一步

请用中文，从独立研究者的角度判断我们当前算子研究下一步该怎么走。下面分别是服务器已经确认的现状、用户提出的想法、尚未决定的问题。请你拿主意，可以质疑我们的思路、建议调整顺序或放弃某种实现。无需沿用服务器agent此前偏好的方案；是否有必要进一步做完整DAG设计，也请你先判断。

仓库：https://github.com/qingyiyi/WAM-egale
算子代码和已测结果固定在commit：319ac6a17daed34e80692796f72a97bf44f4a99f
较新的main用于同步进度，本次算子判断请绑定上述版本。

## 项目希望解决什么

目标模型为冻结WAM/LingBot-VA，投机在action连续去噪分支提出候选，target批量验证。我们希望研究“投机改变小规模action工作负载，进而给Transformer系统调度带来机会”这条主线，但它仍需要实验证据。

算子侧用户明确希望只做完整一个Transformer block：self/cross attention、norm、调制、投影、FFN和残差都在一个persistent CUDA kernel内部执行，切成合适的tile，由常驻CTA/协作warp处理。不同计算可采用不同分块，不要求每阶段都凑满108个CTA。baseline与投机生产代码保持隔离。

## 已经做到了什么

- 硬件为A100-SXM4、SM80、108 SM。当前108个常驻CTA，每CTA8warp/256线程；112384B shared/CTA、255寄存器/线程、静态local1016B、1CTA/SM。这些是编译/函数资源属性，不能直接等同于动态访存瓶颈占比。
- 当前B14×Q16×D3072，24 heads、head_dim128，FFN宽14336；真实短/中/长KV为288/1152/2160。B14来自7候选×2 CFG；线性层展平为M224，attention保留各候选Q16及私有KV可见性。
- 完整单层已放进一个persistent kernel，self/cross attention使用FA2设备计算核心，GEMM复用CUTLASS设备计算核心；多数阶段仍由grid同步分隔，FFN已有四组局部生产消费流水。尚未实现整个block的统一ready任务调度。
- 六个M224、3072→3072投影使用64×128输出tile、8个MMA warp、warp tile32×32、K64/3stage，每投影96个输出tile。Self Q/K/V共用288任务队列，当前按Q96→K96→V96排列；三份权重/输出独立。两段K1536、BF16部分和舍入及bias顺序保持。
- Self Q/K已融合RMS与RoPE。RMS沿每token完整3072维归约，一个128维列块完成还不足以算完整RMS。RMS后先舍入BF16，再按当前源算术执行旋转。
- Warm路径已预计算最终Cross K（含RMS）/V，复用有明确条件边界。历史KV为只读输入，候选private suffix不互相可见。
- FFN1+GELU、FFN2+残差写回融合，FFN1主/尾120个tile、FFN2有96个分段任务/48个最终输出tile；已有四组局部依赖流水，浮点部分和有序提交。
- 当前完整层数学tile为3408/3480/3408，grid同步23次。数学tile、调度节点、CTA、warp不是同一个概念。

## 当前实测结果与不足

同轮、双方同等Cross KV预计算、warm CUDA Graph条件下，相对原Torch静态dense-history路径，完整单层加速比为1.2967/1.8469/2.9910×。这不是最强guarded路径的增量，不是30层、完整chunk或机器人E2E收益。

阶段诊断中，原Torch/自有：Self QKV联合约0.86×，Self输出投影约0.71–0.80×，Cross Q约0.80–0.82×，Cross输出投影约0.74×，LN3约0.61–0.62×，FFN共同区间约1.01–1.02×。阶段参照的缓存/计时边界不同，不能相加预测整层收益。联合QKV队列本身额外收益仅0.08%–0.44%，信号很小。

更大的128×128投影输出tile虽通过有限精度核验，但完整层慢9%–11%，已回退；此前过细ready调度原型也已撤回，负结果保留。最近三个可执行候选各有9真实签名/36结构控制/225次有限完整调用的逐位、有效状态和RNG核验，不代表对所有输入或闭环质量的证明。

远端无GPU硬件计数器或提权权限，可用CUDA Event、软件时间线及编译资源报告。不能假定有NCU新证据。当前GPU正用于RoboTwin既定评估，新的算子方案尚未执行。

## 用户的想法，尚未定案

用户希望以后根据完整任务依赖图安排计算：前驱全部完成、未完成前驱数降到0的任务可以被派发；暂时未就绪的任务等待，空闲执行资源继续做其他合法工作。

用户认为每个计算应先选合适的tile大小和数量，再在某阶段尾部无法填满执行资源时，提前做后续无依赖或依赖已满足的任务，借鉴MegaMoE的tile任务和persistent warp思想。并不要求照搬MoE的结构或固定某组warp数量。

例如，Q/K/V是否可以按同一行块交错派发，尽早释放该块的RMS、RoPE和attention？用户认为这个顺序应留到完整依赖图阶段统一考虑，现在不单独实施QKV交错。

用户进一步问：通用“入度为0就可派发”的执行方式会不会带来很多额外问题？静态调度、动态调度或两者结合，哪一种更适合我们？目前没有选定方案，也没有确定必须删除所有全局同步。

## 请你独立判断

基于这些现状和用户想法，你认为当前最值得做的下一步是什么，为什么？我们应该继续对齐剩余投影/小阶段的性能，还是已经值得开始调度研究？有更合理的切入点也请直接提出。

你怎么看用户的DAG/入度想法？静态、动态或混合调度在这个实际workload上各有什么代价，哪一种值得优先验证？你可以否定通用动态队列、QKV交错或其他设想。请不要为了迎合我们而保证收益。

如果需要更多证据才能决定，请指出最影响判断的未知项和最少应补什么测量。相关论文/官方实现可作为参考，但请核实硬件和问题是否相同，给出直接来源。A100不能默认采用TMA/WGMMA/UMMA。

希望收到你对当前方向、主要矛盾、优先级及下一步行动的判断，方便我们再讨论。输出形式由你决定；先得出判断，再决定是否值得展开实现设计。

## 可核查的源码与证据

- 算子研究/wam_ops/optimized/csrc/whole_layer.cu：完整层、claim_tile、self_qkv_projection_group、norm_rope、action_ffn_pipeline及同步。
- 算子研究/wam_ops/optimized/csrc/cutlass_gemm.cuh：self_qkv8_tile和K分段写回。
- 算子研究/wam_ops/optimized/csrc/fa2_stage.cuh：FA2分组、共享历史和private可见性。
- 算子研究/wam_ops/optimized/whole_layer.py，以及其真实构建/依赖链。
- .pipeline/experiments/single-layer-async-gemm-20261020/small-projection-followup-20261023/STAGE-TABLE.md、RESULT.md、SELECTED.json、CURRENT-STAGE-SUMMARY.json。
- .pipeline/memory/global/requirements/requirement-单层Transformer统一persistent内核调度全部tile.md
- .pipeline/memory/global/requirements/decision-回退阶段全局同步并重新规划tile调度.md

网页若无法读取某份文件，请明确说缺少什么，不要假装已核查。区分实测事实、源码推导、假设和建议；不要改写历史FAILED或科学权限。此次交接只用于独立评议，不授权服务器执行新实验。

上一版交接把问题预先收窄成“请完成DAG设计与指定交付物”，用户已纠正；历史文本保留在commit464bf51243a5ee8c2d07c0f60c9a2be4f7fb1e52，不作为当前要求。
