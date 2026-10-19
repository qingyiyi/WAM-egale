# 新授权：远端无争用工程性能测量

本次用户明确要求速度测试、允许统一远端，并要求速度收口后推进整层warp专职kernel。与旧受阻45chunk入口分开：这是完整自然episode的新计时样本，保留原四项false及原审计失败，不给旧候选补质量证书，不声明独立TEST或非劣。

四套每套10 task，固定init49（case490–499），T/C/P全部新运行；3次独立server进程重复，共360完整episode上限。重复初态是计时重复，不增加原500局成功率的独立样本数。每case沿原Latin顺序，50 action/20 video、tau、TF/normalizer/selector/O3/PF2/优化reset不变；自然成功或horizon800，不截断、不重试、不删长尾。

GPU0顺序Goal/Object；GPU1顺序Spatial/Long。每卡同时只有一个target模型worker，server/client各2 CPU，control各1，GPU UUID与运行来源分别记录。固定源读取，所有新输出~/cxz/performance/libero-four-suite-20261019/isolated；投机源码/旧账本不修改。

新增最多12 GPU生命周期小时（每卡6小时），360局，1452000 target尝试，3GiB证据；每局原1200秒/4000尝试，开局须预留完整局和清理。全链停工作/清理绝对时间见现场REGISTRATION.json，不继承或重置旧M17时钟。失败/启动/等待/超时均计新生命周期。耗尽或实现阻塞则PARTIAL，不自动延长。

每阶段源launcher/independent guard与PID+出生时间证据闭合；外层每5秒核NVML额外compute context，若争用则标污染并在完整episode边界停后续，不杀外部进程。采样不能证明每一微秒绝对独占；记录全部采样。正式计时期间不在同卡运行算子或profile。

报告完整policy推理、action ΣT_round/Σg、自然暖episode、全部尝试时间与双方成功条件性时间比。cold启动单列，不将已完成6000并行质量任务当成这次独占速度数据。所有成功/失败/长尾都保留，速度与成功率并列。

已准备9个三小套件原生阶段（270局）；Long3阶段需独立资源包装，源完整episode数学不变。只有真实READY/进程/终态才报告启动/完成，不把准备代码当结果。
