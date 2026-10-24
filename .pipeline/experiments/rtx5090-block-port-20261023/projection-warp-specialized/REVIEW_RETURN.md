# 四MMA＋一TMA＋三epilogue：有限精度通过，研究采用

评估与当前保留均为 **WARP-SPECIALIZED-4MMA**。原A独立源码/DSO和原TMA负结果保留。

## OBSERVED

仅Self输出、Cross Q、Cross输出三个投影改为W0输入、W1–3后处理、W4–7 MMA；tile64×128/K64，四warp各32×64。两个输入槽full/empty，单结果槽full/empty；MMA最后shared读取后释放输入，下一数值段寄存器计算可与前一段BF16写回重叠。每tile十段9×K320＋K192，原清零/舍入/最后bias次序保留。无六warp、动态寄存器分配或FFN2/DAG改动。

三真实B14/Q16/block0、KV288/1152/2160全部已捕获中间量/输出、有效KV、输入/RNG、normal/debug和角色计数有限逐位通过；默认/非默认stream真实public入口亦通过。每tile各48次A/B、十次BF16提交；四MMA warp各48次消费/释放和十次结果转存，三个epilogue各十次交接。CTA tile内__syncthreads从118次变为3次；不是同步总成本减少同等倍数，仍有局部mbarrier等待。

两独立进程五臂全eager、无Graph、同输入/预计算CrossKV。

|KV|完整层Torch/当前|相对原A耗时下降|相对原TMA耗时下降|Guarded/当前|
|---|---:|---:|---:|---:|
|short|1.2101×|4.843%|5.905%|1.0734×|
|middle|1.7059×|4.447%|5.469%|1.0540×|
|long|2.1253×|3.984%|4.913%|1.0171×|

六个进程/KV观测均优于A，三档平均下降超过事先登记1%投入线；public两流通过，故研究采用。三个投影仍慢于各自原Torch独立区域，比值约0.85–0.88×，不能称每个GEMM均已对齐。完整17阶段本轮重测；插桩扰动约0.07%–0.23%。

资源仍170CTA、256线程、100352B shared、255寄存器/线程、784B静态local；静态相同不证明动态spill/流量相同。

## HYPOTHESIS / UNKNOWN

收益伴随输入/输出角色拆分、同步和warp输出映射同时变化，没有单独证明哪项贡献最大；没有硬件计数器，不报精确TC或DRAM比例。有限三fixture不证明任意输入逐位、完整模型接受路径或机器人质量。四科学权限和production_admitted仍false。

## RECOMMENDATION

先保留这次收益，后续若继续优化，应固定此版本做分量归因；本轮不追加参数搜索。当前因用户要求优先恢复RoboTwin双A100，算子已闭合，暂不新增GPU工作。

源码入口：SELECTED.json；源码快照SELECTED-source；两独立进程结果results/measure1-data与measure2-data；完整表STAGE-TABLE.md；资源RESOURCE-AND-STOP-REPORT.json；14owned身份退出见OWNED-CLEANUP.json。准备时缺少激活环境及一次清理脚本引号错误均为CPU命令失败，GPU调用0，已纠正；构建/GPU阶段均EXITED_OK。
