# 单 chunk 收益为何没有等比例转化为完整任务收益

2026-10-13T10:19:42+08:00（Asia/Shanghai）｜HW chat / research｜本轮只分析已保存证据，GPU 与 target 新增用量均为 0。

**增时发生在完整 policy/chunk 内部，而不是环境执行或 KV 更新。还有一项已定位的共享接入成本：每个 chunk 重做 30 层 Python AST 处理和编译。** 原投机 D 前段较快、后段较慢，抵消了轨迹变短的收益；S0 已带来净 E2E 收益，但重复安装和共同固定阶段使收益小于 verifier 测试。

先校正比较对象：本轮 T=原 target，D=原 K2B2 投机，S=D 加 self-attention 历史共享 S0。D/T 总时间是 +1.05%，接近持平；S/T 已下降 7.55%，不是完全没有变化。S/D 下降 8.52%，而且两者的完整输出、执行命令、接受路径、轨迹和调用计数一致。这是两个初态、单任务的 pilot；三臂分批运行，不把差异都解释为代码的精确因果效应。

**完整任务的账本闭合**

以下为每 episode 均值，单位秒；实际重载六份原始 episode 与六份 CLOSED 请求账本复算。

| 阶段 | T | D | S | D−T | S−D |
|---|---:|---:|---:|---:|---:|
| reset / prompt / CPU T5 | 116.558 | 115.097 | 110.918 | -1.460 | -4.179 |
| 完整 policy/chunk 调用 | 214.034 | 219.761 | 194.179 | +5.727 | -25.582 |
| 观察与 KV cache 更新 | 10.409 | 9.892 | 9.119 | -0.517 | -0.774 |
| 客户端其余时间 | 19.406 | 19.450 | 18.965 | +0.044 | -0.485 |
| 完整 episode | 360.407 | 364.200 | 333.180 | +3.794 | -31.020 |

客户端其余时间 = elapsed − reset − policy − cache，包含环境 reset/5步稳定、env.step、观察提取及客户端逻辑；没有更细计时，不能把它全部称为“纯环境耗时”。EGL 构造/close 的外层 wall 同样给出 S/T 约 1.082×；模型启动、失败启动清理和任务结束后的文件 drain 没有计入主 episode 时间。

两任务合计，D 的 policy 多花 **11.453 秒**，reset 少 2.920 秒，cache 少 1.033 秒，客户端其余多 0.088 秒，净多 7.587 秒。没有看到环境、KV 更新或传输增加到足以抵消收益。客户端与服务端 policy 累计时差仅 T/D/S=0.249/0.268/0.219 秒，两任务总计；这是传输及请求边界外共同开销的账面差，不能全算成网络时间。

**为什么轨迹短了，D 还是没有更快**

T 有 38 个 chunk，D/S 有 37 个，chunk 数只少 2.63%；环境步数少 3.20%。平均完整 chunk 成本却从 T 的 **11.265 秒**变成 D 的 **11.879 秒，增加 5.45%**。按 T 的平均成本做算术分解：少一个 chunk 约省 11.265 秒，37 个较贵 chunk 约多 22.718 秒，净多 11.453 秒。这是平均值恒等式，不是同状态 chunk 的因果实验。

逐 chunk 记录显示，早期加速不能外推到整个 episode。为避免终局单 worker 尾段干扰，下表仅比较两初态共有的前18个位置；平均值来自服务端原始请求 wall，包含完整 video、action、输出处理。

| 轨迹位置 | T 秒/chunk | D 秒/chunk | S 秒/chunk | D/T 耗时变化 | S/T 耗时变化 |
|---|---:|---:|---:|---:|---:|
| 前段 chunk 1–7 | 10.931 | 10.312 | 10.107 | -5.66% | -7.54% |
| 中段 chunk 8–13 | 11.213 | 11.820 | 10.437 | +5.41% | -6.92% |
| 后段 chunk 14–18 | 11.672 | 14.639 | 11.533 | +25.43% | -1.19% |

这些是各自实际 rollout 的描述性曲线；T 与 D/S 的状态不同，位置也同时关联 KV、接受路径和并发，不证明 KV 是唯一原因。此前同状态固定负载 OFF 测试则直接验证过：短 KV 双需求动作分支快 14.9%–15.3%，长 KV 双需求慢 28.2%–30.2%。这与当前前快后慢一致；旧实验没有 S0，不能冒充当前 S 的同状态测量。

![逐 chunk 曲线与任务阶段耗时](/rjs/cxz/论文探索/.pipeline/research/speculative-sampling/evidence/speculative-e2e-attribution-20261013/E2E-ATTRIBUTION.png)

图的左侧使用两初态共有位置的 server wall，右侧使用全部完整 episode 的 client wall；两者口径标明，未混合成同一阶段计时。T 与 D/S 不是逐位置同状态对照。

**验证次数少了，验证工作未按比例减少**

T 每次普通 action forward 是 CFG batch2，D/S 每棵 K2B2 树验证 7 个候选、CFG batch14。两任务 action 分支的前向次数 T=1938、D/S=1122，但累计 CFG 行调用 T=3876、D/S=14556，即 **3.755 倍**。总 target 前向次数少 29.88% 不能当成总计算量少 29.88%；行量也不是精确 FLOPs 或 wall time。

D/S 两任务构建 1026 棵树、执行 3078 次 draft forward、6626 次树 scheduler transition，最终提交 1850 次 denoise 更新。候选打包、分支投影、误差打分、选择/回退会产生额外 CPU 调度和同步。S0 只避免候选历史 K/V 的重复物化/读取，当前 Q/K/V、cross-attention、FFN、video、VAE 仍沿用原计算，prompt/cross-KV 与行证明复用仍关闭。

旧固定负载诊断 ON 中，verifier forward 占 action 69.3%–83.7%，draft model 占 2.2%–4.7%。这支持优先处理验证器成本，而不是先训练或替换 draft。ON 有同步观察扰动，且不是本轮逐阶段计时，不能用这两个范围给当前 E2E 分摊精确秒数；嵌套 tree_build 与 draft/scheduler 不能重复相加。

**共享版新增的明确成本：每个 chunk 重复编译安装**

源码链条：OnlineShared.action 每个 chunk 新建 SharedPrefix 并调用 install；父类 SingleWorkerCHC.install 对30层逐一调用 _private_forward。每次 _private_forward 都重新读取/解析原 model.py、deepcopy AST、校验 AST、compile/exec，最后绑定当前适配器的回调。当前 37 个共享请求都会走这条安装路径，源码推算为 1110 次 AST 构造；原始 E2E 未另加 AST 次数计数器。

| 已核查的范围 | 数据 | 可以得出的结论 |
|---|---|---|
| 当前 D/S 被动日志：chunk 入口至第22个 FRAME START（首次普通动作准备） | D 130.753s；S 170.025s；S 多 39.272s，平均 1.061s/chunk | 共享版在动作入口之前新增了明显主机成本 |
| 同一窗口之后至原 policy 返回 | D 308.500s；S 218.113s；S 少 90.387s | 其后动作/输出段确实更快，前段增时抵消部分收益 |
| 当前主机、真实生产类30层 CPU 安装诊断，6次，无模型 forward | install 中位 1.276s，其中 AST 1.247s，占 97.72% | 重复 AST 编译本身足以产生每 chunk 约一秒的主机开销 |

被动日志窗口包括 video、前置处理和 S 的安装，并非精确 install profiler；两个阶段分批且 CPU 诊断使用随机 tiny 权重，1.276 秒不是 GPU 模型性能，也不能把全部 39.272 秒严格记为 install。**源码、CPU 重现、原始日志三者形成一致证据，重复安装是优先消除的具体成本。** CPU 诊断调用原安装函数，30层原方法与 KV 完全恢复，0模型前向、0 CUDA 初始化；没有改原实验或生产源码。

先前 verifier replay 的 control.install 在计时外；sequence 中冷 lease、staging、key、release 在计时内。“cold setup included”只覆盖 lease 序列的设置，不包含适配器安装。完整在线 chunk 则包含每次安装。先前所谓“单 chunk”须区分：固定 action 分支与 verifier 序列都不覆盖完整 chunk/episode。最新 replay 短/中/长 KV 的 S0 中位耗时下降 17.49%/23.42%/34.16%，也来自单 worker；短档5/6胜出仍为mixed，中/长为positive pilot。当前 episode 是双独立 worker 同卡并行，不能直接沿用那个比例，也不能把中位数改善写成每个chunk稳定加速。

S/D 日志的入口窗口多 39.272 秒，入口后窗口少 90.387 秒，净 policy server 少 51.115 秒。也就是说，看动作入口后的快段会高估净 chunk 收益。窗口是毫秒精度的主机日志重建，涵盖原保存/CPU copy/empty_cache；这些子项仍未在本轮单独计时。

**固定阶段继续稀释 E2E 收益**

reset/prompt/CPU T5 每 episode 约111–117秒，占 T 完整时间32.34%，S0 没有优化它。D→S 完整 policy 实测下降11.64%，policy占 D 总时间60.34%；只让 policy 发生这一变化，其他阶段保持 D 原值，算术上只带来约7.02%的 E2E 降时。这是固定耗时的直接稀释作用。

本次 S/T 的实际7.55%中，还包含非共享阶段 reset 的跨批变化。把 reset 算术统一成 T 的耗时，S/T 降时是 **5.99%**，仍正向；这不是新测试或精确因果收益，只是避免把 T5 波动归功于共享。原报告7.55%作为观察到的任务结果保留，不改历史。

**下一步：先消除重复编译，暂缓轨迹控制**

建议先做私有的“编译模板复用”：按冻结源码 SHA/原 forward 身份缓存不可变的 AST 编译结果，在进程内复用；每个 chunk 仍重新绑定当前 callback/owner，并独立建立、撤销 lease。不能跨请求复用带 model/KV/旧 callback 的函数，也不能跳过源码身份、30层准入、CFG 行、cache generation、版本和异常失效检查。首 chunk 的冷编译也应计入完整 chunk 时间，不能通过移到计时外制造收益。

先 CPU 验证原有所有失效/线程/恢复约束、完整输出及安装成本；然后以新有限方案做同状态短/中/长 KV、同卡双 worker 的 T/D/S/模板复用 S 对照。正式性能使用 profiler OFF，至少分别记完整 chunk、video、action入口设置、action、后处理，诊断 ON 单独运行；完整30通道动作、video、KV元数据、scheduler和RNG保留逐位门。通过后再复测完整 episode。

如果这项接入成本消除后长段仍无净收益，再拆 batch14 验证器内 self/cross attention、FFN、打包与同步的成本，研究计算复用或成本准入。reset/T5 缓存可作为后续三臂共同工程基线，但不能记作投机专属收益。当前不改阈值、树策略或环境轨迹，不重新训练 draft；原轨迹控制方案保存，优先级按用户最新要求后移。

本轮完成的是原因分析与下一步设计，没有声称已实现或验证新的 GPU 优化。新 GPU/target 用量=0；累计仍21489 intents、6545.956473秒/18000秒。protected state/rules/cycle、公共代码及旧结果由归档核验，HW仍为研究追加通道，不重开M04。

**可追溯证据**

- [本轮复算与完整逐 chunk 数据](../../../review/OMISSIONS.md)；[CPU安装诊断](../../../review/OMISSIONS.md)；[分析工作清单](../../../review/OMISSIONS.md)。
- [原完整episode结果](SPECULATIVE-E2E-RESULT-20261013.md)；[旧同状态负载矩阵](FIXED-LOAD-COST-RESULT-20261011-213036.md)。
- [每chunk安装入口](../../../投机采样/experiments/speculative_e2e_v3/online_shared.py)；[30层安装循环](../../../投机采样/experiments/single_worker_chc_gpu_v4/adapter.py)；[AST处理与编译](../../../投机采样/experiments/single_worker_chc_gpu_v4/adapter.py)；[replay计时边界](../../../投机采样/experiments/shared_prefix_gpu_v3/worker.py)。
- [原始action候选验证与计数](../../../投机采样/wam-p0-worktree/wan_va/wan_va_server.py)；[CPU复制后异步文件保存](../../../投机采样/wam-p0-worktree/wan_va/utils/utils.py)。
