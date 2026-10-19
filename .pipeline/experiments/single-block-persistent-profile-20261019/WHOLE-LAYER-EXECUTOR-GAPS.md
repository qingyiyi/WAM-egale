# 单层统一 persistent kernel：目标与实现缺口

OBSERVED：M32没有完成用户要求的整层统一persistent kernel。本轮M33是该局部原型的退化profile，不把新增profile工具或216 CTA实验变体作为整层实现。

## 用户要求与完成标准

一次完整单层调用，由一个常驻GPU计算kernel执行全部self-attention、cross-attention、FFN及其norm、调制、RoPE、投影、残差和合法KV工作。host只准备参数、空间、任务描述并启动；不能在中途调用cuBLASLt或再launch多个计算kernel，然后因为只有一个C++入口就声称实现完成。空间初始化、数据搬运等边界单独登记并计入总成本。

只做一层。当前输入[B,Q,3072]、context[B,512,3072]，H24/d128、FFN14336，输出与输入同形。B2/B14 action Q16对应GEMM M32/224；每个逻辑attention实例Q16，候选private KV隔离。B2 video Q128是必要兼容输入，不扩成30层。

## 当前实际覆盖

| 部分 | 当前源码 | 设备tile/常驻情况 | 尚缺 |
|---|---|---|---|
| 整层入口 | optimized/csrc/block.cu:317–338 | host按self→cross→FFN顺序提交 | 单个设备执行循环 |
| Self/Cross attention | attention_tiles.cu:42–74 | Flash设备tile；各阶段固定步长CTA循环 | 纳入统一ready任务与workspace协议 |
| FFN1+GELU | ffn_tiles.cu:51–74 | CUTLASS设备tile；保留BF16线性输出再GELU舍入 | 跨阶段任务入口与可复用storage |
| Self Q/K/V/Wo、Cross Q/K/V/Wo | block.cu:276–294 | host linear→cuBLASLt | 8次设备GEMM的原数值契约 |
| FFN2 | block.cu:313 | host cuBLASLt | 设备GEMM；旧M26逐位失败不能继承FFN1通过结论 |
| norm/调制/RoPE/残差 | block.cu及norm.cuh | 各自独立global kernel | 显式tile/线程布局的device入口 |
| KV布局/私有写/pack/恢复 | block.cu:241–260 | 多kernel，含16B D2H与stream同步 | 设备端合法视图和错误状态；不能隐藏在host同步里 |
| 调度 | persistent.cuh:7–17 | task=CTA编号+k×grid；4warp一起做 | 整层依赖、ready发布/获取、任务代际与终止 |

## 第一版执行组织（设计，未实现）

```text
single_layer_persistent<<<resident_CTAs, cooperating_warps>>>(inputs, weights, workspace, task_state)
    while 整层未完成:
        领取一个已经就绪的任务
        warp组协作执行该任务的device tile
        完成异步拷贝、必要CTA同步
        发布输出和完成计数，释放满足依赖的后继任务
```

常驻CTA是SM80资源容器，内部warp组执行tile。当前Flash/CUTLASS原语使用4个warp与CTA barrier，不能让一个warp独自调用后让其他warp跳过barrier。第一版全组协作执行任务，后续再研究加载/计算/调度专职warp；不照搬SM100的TMA/UMMA/TMEM实现。

任务类型按Transformer DAG建立：mod-table，norm/mod，GEMM，Q/K RMSNorm，RoPE，KV管理，attention，residual。GEMM任务是一块输出矩阵并完成其完整K归约；attention任务是一块query输出，内部依次QK→online softmax→PV，保持原KV遍历/归约/舍入。首版不拆出全局score矩阵，也不任意改变split-K。

关键依赖：

1. norm/RMSNorm必须等对应行的全部3072列就绪，不能按单head128维独立归一化。
2. self attention等待合法当前候选的全部可见私有KV；非因果Q16不能只等同一query行。
3. Wo等待对应行24个head的完整输出；FFN2等待14336维activation完成。
4. cross context K/V可在条件已定时与self支路准备并行，但cached/uncached分开，已有缓存不是新贡献。
5. 发布输出采用release/获取采用acquire的设备同步协议；完成计数带generation，避免旧任务就绪污染新调用。仅已就绪任务入队，不能所有常驻资源都占着等待尚未运行的producer。
6. 首版各中间量有固定不重叠workspace，验证生命周期后再研究复用；跨CTA普通shared memory不可共享。队列有界、剩余任务计数和错误终止必须明确。

## 推进顺序与门

M34.1：实现并验证缺失device原语，优先量化QKV/Wo/FFN2与原cuBLASLt的真实中间结果差异，同时准备norm/RoPE/KV设备接口。不是继续把host linear包进函数。现有M26 FFN2六点最大绝对差0.25，禁止降低门或当动作误差。复用Torch的host包装不能解决封闭cuBLASLt内核的设备可调用问题。

M34.2：实现单层统一任务表、ready计数和常驻执行循环；先完整覆盖DAG和无死锁，再做跨子图流水/warp角色优化。分阶段原语验证是开发步骤，只有所有阶段在同一个计算kernel内才满足最终结构目标。

M34.3：真实B2/B14短/中/长及video完整输出、原中间量、全部有效KV、输入/RNG/参数/API；检查trace计算launch边界。失败则保留独立研究实现，不替换baseline或生产投机。有限fixture通过不代表所有输入逐位等价。

M34.4：同输入比较冻结baseline、强B14 shared-prefix/cache/Graph适用路径与统一kernel，分别计设备时间、完整block、工作区和资源。统一kernel若按最大寄存器/shared-memory需求分配，可能降低驻留；必须测，不以kernel少直接宣称加速。

M34是下一实现任务；本页不是已经实现、编译或GPU验证的声明。新增作业先登记实际预算与停止条件，不能用M33余额自动扩大网格/实现搜索。

## 本次解释的失效关系

旧M30设计允许保留bulk host GEMM的阶段，仅可继续用作开发对照，不能满足此次整层统一kernel目标。旧30层门已被单层要求替代。旧M32有限逐位和性能负结果不改写。
