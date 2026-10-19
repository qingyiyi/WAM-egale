---
kind: execution
cycle: wam-research-20261016
updated: '2026-10-19T14:49:18.383103+08:00'
---
# 最近执行检查点

## 2026-10-19T15:13:41.358748+08:00：完整native首轮六点过

M28新顺序C++/CUDA包含Norm/RoPE/投影/self-cross/FFN/残差/KV，无ATen高层数学转调；GEMM直接cuBLASLt复现原配置，独立attention为自有CUDA。共享GPU仅做精度，六真实block0每组23关键中间量/输出与六原语全部逐位过；有效KV/metadata/输入/RNG/27参数不变。FFN2同query直接GEMM亦过，旧CUTLASS失败不改。首次build1命名冲突失败保留，build2过。默认尚未切换；继续写缓存/驱逐/Q128与30层门，性能待自然空闲。[首轮](../../experiments/block-native-baseline-20261019/CHECKPOINT-1.md)。


## 2026-10-19T14:49:18.383103+08:00：短KV/B14核查与比较方向澄清

用户确认关注自有实现比Torch慢；继续询问2.5%含义和源码/完整Transformer覆盖。M27分列默认复现/Torch与FFN融合/同配置未融合；后者约3%为更快。源码绝对路径、默认仅SDPA及可选FFN1边界已解释并通过Maintain合并进可读记忆。

新增A/A真实重放两次foreign GPU中断、一次有限等待到期，无自动扩展；665.685秒生命周期含等待/失败/清理，GPU阶段墙钟31.145秒非CUDA active，6180attention/918block intents，0full30/机器人episode。三个owned实际退出，无他人进程被清理，旧PARTIAL及所有样本保留。短/中B14同等cached Graph比约0.999/1.001，不把未全矩阵包装通过。

只增加共享诊断工具、CPU预读参数及文档；baseline、1875投机文件、保护文件/原tile不改。未找到稳定的新增数学开销证据；未盲改CUDA内核。旧M22慢点、M25严格筛查、M26完整FFN2失败及NCU偏差仍在。[完整结果](../../experiments/operator-shortkv-b14-20261019/RESULT.md)。

原Object巡检06:46:38UTC：{'T': 426, 'C': 227, 'P': 227}，3活动/自动后继，无terminal；原M17预算不变。

原执行归档：[历史](history/M27-before-investigation-20261019/EXECUTION.md)。


## 2026-10-19T15:36:26.606514+08:00：完整native接入研究入口

baseline六真实action逐位过；原生optimized亦六点严格过。12个连续write1/write2/read0状态门过含long驱逐。旧视频文件缺完整block_args，原attempt1失败保留；新独立完整canonical policy捕获已启动，仅已有冻结请求，不是机器人episode/质量/计时。研究baseline/optimized入口已接native，不改投机生产。所有新GPU作业仍M28原3600秒/1GiB/22:54清理内。
