# WAM短状态入口

当前Cycle：wam-research-20261016；M39单层算子、M40 RoboTwin、H02本机轻量化。2026-10-25 01:05更新。旧state/cycle/rules、原tile、FAILED及科学门保护不变。

M39新附件T0–T3已收口NO_GO：基线当前SM120 SELF-QKV-TMA-4MMA，来源3eb0026705e603d6268731bb26b34261acca696c且现场源字节相同。唯一PAIR2048x2将相邻K2048段同CTA、672→384 owner，ACT仍global、FFN1不重算、七次BF16舍入不变。三真实block0 fixture七段P/R、27捕获项/输出、输入/有效KV/RNG有限逐位通过，但两个独立进程三KV无插桩整层慢6.25%–7.89%、同轮FFN慢18%–19%，不采用，当前A入口/DSO未改。

当前同轮原Torch/A完整单层KV288/1152/2160约1.251/1.758–1.761/2.191–2.201×；Guarded/A约1.110/1.084–1.089/1.047×。三方5090/B14/Q16/eager/no Graph/warm CrossKV，单层真实重放，不是在线verifier或机器人E2E。完整17阶段表按T0/T1/T2/T3在聊天打印并保存。T1前序等待约0.2μs/任务，MMA/输入与结果交接包络25.8μs，非关键路径占比；新旧资源相同，不支持shared增大归因。DRAM/L2/动态spill未知。

本轮5214完整block、job生命周期合计452.531秒，编译409.888秒、GPU工作进程42.384秒、CUDA active未知；2次GPU前准备失败、1次定位修订，0 CUDA编译/数值失败、0 timeout。13phase与5guard核验退出，5090 context为空，实例仍开。现有5090持续使用授权有效，逐job有界，不扩A100。上轮TMA输入/跨任务epilogue/M16驻留/双CTA DSM四负结果保留，详见NEGATIVE-EVIDENCE，不重复运行或改名。

M40 01:05现场：405/600完整组三臂、1215/1800局，Target/TF/PF2成功326/328/332（同分母405）。两张A100 worker身份存活、RUNNING，队列HALT=null，自动接未开始组。00:53 pair406/T观察VAE编码显存OOM，旧PARTIAL及同组C/P自然失败保留；新GPU1 guard只加模型前24GiB/完整局前6GiB空闲准入，数学及reset bank不变。406/T未自动重试，旧pair281/C一次授权尾部恢复仍会接续；缺406/T不能报告600完整组。原160GPU小时累计约86.15，原T0与更早10/29 14:03 guard截止保留。微波炉66/100组，等100组三臂齐全再查因。并行墙钟不是正式E2E加速。

H02：TMPDIR/TMP/TEMP统一/rjs/cxz/tmp；623项闲置/tmp归档清除、28活动项保留，模型公共组件逐字节核验去重释放13.2068GiB。25564大文件冷归档到BAAI ~/cxz/backups独立目录，传输校验/fsync确认后逐文件或闭合小批立即释放本地，每30秒落状态收据。原22:21起点累计12小时、10/25 10:21:19截止不变；实际释放看BULK-STATUS，不把预计120.62GiB当完成。恢复入口保留，不删唯一失败证据/代码/原tile/必要fixture。

质量边界：TF64为研究载体，旧TF_ON连续NO-GO、72/96 PARTIAL及三风险失败保留；不拼旧T/DEV/新50为独立TEST。PF2不同于bonus，不恢复物理状态。engineering_quality_screen=false、independent_test_complete=false、noninferiority_established=false、formal_performance_permission=false、production_admitted=false。新算子未扩大验证至其他层/在线接受/完整模型/机器人闭环。

下一步：保留当前大M与global ACT，先解释FFN2 MMA/输入交接包络，再决定新候选；本轮停止实现，不自动展开DAG、warp比例或FA2重写。baseline/投机/原tile不改，不做smoke/独立模型SHA，不杀他人进程或改驱动/服务/时钟；BAAI只写~/cxz。论文中文由用户飞书整理，不新建论文草稿。11/01 18:00中文证据包内部节点，主张随证据收缩。

- [完整进度](cycles/wam-research-20261016/PROGRESS.md)
- [本轮结果](experiments/sm120-ffn-next-20261024/REVIEW_RETURN.md)
- [当前完整阶段表](experiments/sm120-ffn-next-20261024/STAGE-TABLE.md)
- [固定源与原始数据](experiments/sm120-ffn-next-20261024/SOURCE-AND-RESULT-ENTRY.json)
- [RoboTwin现场/接续](experiments/robotwin-pilot-20261020/eval100-20261022/external-oom-20261025/PLAN.md)
- [存储状态与恢复](maintenance/storage-placement-20261024/PLAN.md)
