# M32：远端tile与persistent实现

用户已明确同意远端实现、先tile/persistent、完成后汇报；最终E2E留本机。新作业按REGISTRATION.json计账，不能挪用已结束M31预算。

- [x] 读取M31、冻结88项baseline/依赖，核查GPU0与GPU1归属。
- [x] 实现独立optimized的stage-local persistent CTA；保持sequential/tiled模式。
- [x] 在~/cxz隔离工具链、源码、九个真实block输入与产物；复用远端既有target权重。
- [x] CUDA12.4/SM80编译完成；Torch2.9.0+cu126，不改全局环境。
- [x] 九个真实完整block与90项连续有效KV状态逐位通过；输入/RNG/参数/API及中间量门通过。
- [x] 真实self/cross QKV与FFN1共48项输出/逐tile覆盖通过，默认CTA与2 CTA压力路径均核验；实际kernel名称通过。
- [x] 两独立进程九点完整单层block探索计时完成；persistent没有新增加速，不报完整模型E2E。
- [x] 用户收缩范围：取消本轮30层迁移与重放，当前只验完整单层block；取消不是验证通过。
- [x] 结果/失败/资源/owned清理归档，HW完成；main镜像由发布收据单独确认。

## 具体实现

attention：task=(candidate/CFG batch, head, query block, legal KV split)。保持Q16/Q128、原可见性、Flash在线softmax以及split/combine顺序。固定CTA网格按task_id=cta_id+n*grid_size分配任务，各CTA的warp共同计算tile；默认CTA数不超过SM数量。没有跨CTA等待或原子领取竞争。

FFN1+GELU：task=(row tile, output-column tile)，保持原SM80 CUTLASS MMA和BF16 Linear→GELU舍入。完整tile后等待cp.async完成并CTA同步，再复用共享内存。FFN2仍cuBLASLt。

两个primitive以现有完整self/cross/FFN子图接入，同stream保存完整阶段依赖。没有跨阶段ready队列、warp生产者/消费者专职分工或跨候选shared-history新复用；这些不是本次已经实现的结果。persistent不默认带来加速。

## 对照、预算与停止

先正确性，GPU1上的Object继续；按GPU0 UUID检查资源，CPU固定两核。源码/输入/编译迁移最多5GiB、研究证据512MiB；GPU生命周期30分钟、900 block与0完整30层forward上限，3小时绝对墙钟，清理提前5分钟。准备、传输失败和等待均记墙钟。

baseline源码和本机冻结DSO不变；其DSO存在本机绝对依赖路径，远端不改写或伪装成相同产物。远端本轮以原Torch及独立sequential/tiled为oracle和调度对照，baseline不加载。结果中明确此范围。

网络或环境不满足就保留PARTIAL，不伪造GPU结果。数值/状态门失败立即停止对应集成，不放宽标准。本机最终E2E、最强shared-prefix/O3/Graph和接受/完整机器人路径不在此次有限重放结论内；四项false保持。

## 正确性通过后的单层计时登记（2026-10-19 20:55，北京时间）

不扩增原预算。仅GPU0无其他进程时执行两个独立进程，每进程252次完整block调用（九输入、七入口、一次检查warmup、三轮旋转顺序）。入口为original_A / sequential / tiled / tiled_ffn / persistent / persistent_ffn / original_B。每次恢复相同有效KV，计时包含真实动态block的KV管理/打包/同步；安装参数与cache恢复不在窗口内。GPU event与host墙钟均保留，三样本中位只是探索性测量，A/A和原始值全部发布，不用微小差异宣称稳定加速。

CPU4/5、线程2、每进程480秒硬限；累计预计731次block调用（已有227＋计时504）≤900，已有生命周期31.722秒，新两进程仍受原1800秒总限及22:47停工作/22:52清理约束。另一张GPU继续Object，主机仍有共享负载，因此本次是同GPU单层探索计时，不是正式E2E或最强verifier计时。遇新GPU0进程、正确性失败或超时立即终止计时链，不自动重跑。

计时attempt1在0次block调用时因NVML宿主PID与容器PID不同退出；原记录保留，计入12.781秒生命周期。修正仅计时资源守卫：启动前GPU0无context，本worker初始化后必须唯一新context，随后进程集合严格不变；这是现场关联，不宣称已证明宿主PID映射。新attempt2/3接续既定两个计时进程，无算术变更，不扩预算。
