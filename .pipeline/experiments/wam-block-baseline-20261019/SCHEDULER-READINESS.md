# 是否可开始tile scheduler

结论：**尚不可开始收益验证型persistent scheduler**。完整host顺序B0的CPU基线已建立；GPU数值、状态、A/A环境与强baseline配对时间尚缺。

| 项目 | 当前状态 | 缺口 |
|---|---|---|
| 原SDPA的独立CUDA kernel | EXISTS，旧有限GPU通过 | host入口含ATen分配/launch；vendor存在CTA绑定的device模板，但未形成调度器可组合ABI/资源证书 |
| shared/guarded历史attention | EXISTS，有CTA绑定device模板及有限旧输入通过 | 固定128线程/extern SMEM/同步；新联合cache/block仍须门，非可随意调用的全block原语 |
| GEMM | 原Torch/cuBLAS host接口 | 无已对齐且不明显慢的D0设备tile primitive |
| norm/GELU/row ops | 原Torch host接口 | 无新设备调用约定、资源/舍入证书 |
| P_seq/P_sched | NOT_IMPLEMENTED | 同一D0、同数学配置后才比较调度增量 |
| 完整block调度DAG | CPU阶段已列依赖 | 无真实GPU profile占比、ready粒度/同步/寄存器/SMEM测量 |

唯一下一优先：先让新B0在六真实fixture通过GPU数值/有效状态门与A/A/完整block配对计时；噪声失败先解决可比较性。通过后再只对实际热点选择一个D0，例如FFN1→BF16 GELU→FFN2行组链；目前只是备选，未按FLOPs猜成最大时间热点。

条件触发设计：静态任务结构、轻量ready计数、CTA领取M/N输出tile；第一版不加跨CTA split-K，必要global scratch可保留。LayerNorm需完整hidden行，FFN2需完整activation K reduction，必须保留BF16中间舍入。更多candidate行不自动提供任意跨层/跨token独立性；A100没有Hopper TMA/WGMMA。

潜在收益上界只能用实测份额f算Amdahl，例如理想上界1/(1-f)；当前f未知，不给虚构倍数。重写GEMM损失权重复用、Tensor Core效率或改变累加顺序可吞掉launch收益。P_sched>P_seq才支持调度增量；仍慢于R_strong不能宣称系统胜出。

借鉴[CUTLASS4.2.1 grouped scheduler](https://docs.nvidia.com/cutlass/4.2.1/media/docs/cpp/grouped_scheduler.html)仅指任务领取设计；它不是完整WAM依赖DAG实现。本轮无D0/P_seq/P_sched代码、无30层fusion、无GPU调度实验。
