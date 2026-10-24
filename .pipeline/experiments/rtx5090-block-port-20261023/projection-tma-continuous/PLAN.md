# 三个投影的跨分段连续TMA输入流水

用户明确同意先试连续TMA输入，后处理warp重叠另行讨论。保持96个输出tile、每tile十段（9×K320＋K192）、8个MMA warp、原BF16累积/bias顺序。两K64输入槽持续预取，跨段不重启输入ring；每段仍独立FP32部分和，从零开始。输入与输出shared不可union别名。

- [x] PREP：读取当前契约/账本，冻结当前FFN2-SPLITK为独立A。
- [x] IMPLEMENT：复用原CUTLASS warp MMA/layout，TMA两槽＋寄存器fragment预取；不改FFN2/其他阶段/树/生产。
- [x] EXACT：三真实B14 fixture的26中间量＋输出、normal/debug、全部有效KV、输入/RNG；失败不计时，最多一次有依据修复。
- [x] MEASURE：两独立进程完整层Torch/guarded/A/候选，全部eager无Graph、同预计算CrossKV；诊断17阶段另列，采用以完整层为准。
- [x] CLOSE：默认/非默认public验证、保留或回退、完整阶段表直接打印与保存、回传/owned清理和预算终态。

上限50分钟含准备/编译/等待/失败/清理，原累计32386.4125/36000秒不重置；预留300秒清理。GPU仅现有5090单worker，CPU4，新存储1GiB、完整block15000。先不加入新warp分工、跨CTA拆K或其他tile候选；不改A100队列、baseline、原tile/投机和保护文件。候选严格有限正确性通过且完整层稳定收益才研究采用，否则恢复A。四科学权限及production_admitted保持false。

TMA发起由现有MMA线程中的一个线程负责，无新增专职warp；all-CTA在复用输入槽前完成读，在TMA完成后才能读。global generic写入在投影入口用async proxy fence发布；full barrier phase按同slot使用次数推进。未保存全部输入案例，不声称任意输入等价。

候选两进程完整层退化1.065%–1.384%，停止采用。恢复A后补本轮当前17阶段及实际public两流核验，均在原调用/生命周期预算内；不是新优化候选。

终态：有限逐位通过，完整层退化1.065%–1.384%，未采用；A已恢复、当前全表/两流/回传清理闭合。
