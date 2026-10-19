# 新计时边界：任务常驻、推理与冷启动分列

读取实际冻结源码后登记，没有改动计时实现或已有字段。

| 指标 | 实际包含 | 实际排除/限制 |
|---|---|---|
| `ΣT_round/Σg` | 原action_total完整区间的CUDA events；draft/pack/verify/accept/fallback/bonus与terminal归属；每chunk合法50次scheduler更新 | video、reset/cache RPC、环境动作；g不是环境步数 |
| `policy_server_seconds/chunks` | 同步返回的完整原policy推理host时间，含video/action及原infer内部工作 | 独立reset/cache RPC、环境动作 |
| `policy_RPC_seconds` | 客户端每个实际model.infer到返回的wall，含网络/序列化 | 环境动作、其他RPC |
| `episode_sec` | 环境已经构建后、reset前开始；初始化/稳定、reset bank、policy/cache RPC、动作后处理与完整自然环境执行 | 模型进程启动、环境构建、finally env.close、后续证据写盘 |
| `client_total_wall_sec` | 入口main起到终态写出前，含大部分导入、环境构建、env.close、raw证据落盘 | 先前已常驻的模型加载；并非纯机器人E2E |
| `MODEL-SETUP-OBSERVATION` | 一次server启动到真实健康READY；原生代码用monotonic和实际birth ticks | T/C/P共用一次，不能任意分摊成三个独立cold时间 |

源位置：`投机采样/scripts/p0_libero_multichunk_client.py` 328/345/375/507/524行；`tf_suite_pipeline_v1/client_entry.py` 189/239–252行；Long相同完整循环。本文行号是核查时定位，固定代码在当前公开源中可追溯。

`tf_fullsuite_eval_v2/evidence.py` 244–246行明确最后round覆盖terminal调用；`role_contract.py` 36–47/72–85行核对T为50合法更新、51 CFG2 action调用（包含1终端）。S/T均覆盖50更新，不能删第51调用来补成本。episode_COUNTER核对`g=50*policy_chunks`，timing rounds不重复相加。

full policy与action区间嵌套，不能相加。`episode − reset_RPC − policy_RPC − cache_RPC`只能称**客户端非RPC剩余时间**，含模拟/预处理/审计等，不全是环境动作。`policy_server − action_cuda`也是混合残差，不能全归给video或VAE。`major_host_spans`虽然在内存收集，当前持久化COUNTER未保存；没有完整原字段，逐video/obs/VAE精确拆解为UNKNOWN。

原COUNTER的`exploratory AutoDL`/`performance_claim=false`文字原样保留；本轮实际物理卡/独占观察以新登记、GPU UUID与队列原采样为准。这里报告同卡工程成本，不改变旧质量四false、不将重复init49当独立TEST。

自然任务全部失败时间保留，并报告双方成功条件性比值、实际步数/800罚分、每次独立进程重复。更快失败不作为计算收益。成功episodes/Σepisode-hour是暖任务有效时间口径，不等同含启动/等待/cleanup的GPU租用小时；后者算法逐臂冷吞吐尚未独立测量。
# 额外审计开销边界

新计时沿用原完整episode观测器。`tf_fullsuite_eval_v2/controller.py:67–80`中，`original_infer_host_seconds`从`evidence.begin`之后开始，在原infer和显式返回同步后停止；后续`evidence.finish`的首两S完整状态比较不在该server policy定时内，但仍在client RPC和warm episode内。infer内部的事件/计数观测开销仍在对应区间中，没有另测关闭审计的生产部署版本。

因此policy RPC减server policy时间是包含wrapper审计、传输/序列化等的混合残余，不能全部叫网络或仿真成本；warm也不是去掉审计后的纯部署E2E。action CUDA与policy host使用不同但嵌套的时钟边界，差值只作混合残余，不直接命名为video/VAE阶段。原`performance_claim=false`仍保留。
