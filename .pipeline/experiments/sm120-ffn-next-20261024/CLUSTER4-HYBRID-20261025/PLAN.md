# 四CTA与global混合FFN有限验证

T0 2026-10-25T13:49:29.799972+08:00；硬截止 2026-10-25T14:49:29.799972+08:00。预算含准备/编译/等待/失败/清理：60分钟、10000整层调用、4CPU、1GiB新增证据。

- [x] T0 固定A、源码/工作量/资源与进度证明。
- [x] T1 B4仅cluster4启动；C4保留A FFN1/globalACT，FFN2四CTA分担不同N128输出，共享四个M64×H128片。
- [x] T2 真实三KV全中间量/七段/状态核对；失败先停，不改数值标准。
- [x] T3 同轮eager原Torch/Guarded/A/B4/C4无插桩整层；独立profile标明边界，不求阶段和。
- [x] T4 结果、完整17阶段快照、资源/归档与采用决定。

C4：168个cluster job，每job四CTA、一CTA一个M64×N128×K2048输出。FFN1仍120任务、FFN2仍672分段任务；每job四次K512搬运，每CTA只存M64×H128=16KiB激活，另存局部A/W2双缓冲。K512内按K32原序MMA，七段BF16提交不变。每个K512边界仅cluster内同步，原层间grid同步保留。ACT先global，局部共享只减少重复ACT读取请求，不声称减少HBM；跨cluster仍global。

进度：cluster job按partition-major静态递增，每个predecessor job id更小，全部cooperative CTA驻留，FFN1有限任务先完成并release；只等已拥有更小job的组，不产生等待未领取后继的环。最后cluster同步后才能覆盖DSM源slot。

旧DSM8/R1负结果不变；本次最新用户4CTA授权另立有界job，不重置旧账本。本机A100暂停、远端双A100继续。
