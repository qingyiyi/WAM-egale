# 冻结 TF_ON 连续投机 / PF2 完整 DEV

独立执行 namespace，科研父版本为 `4f04b35e44cbc00483dc6245ad57abec718385c9`。
不覆盖旧 campaign、FAILED、retention 或生产默认。实现提交另外登记；
本目录存在不代表实验已完成，也不提供任何质量或正式性能放行。

两个 arm 共用一套 wrapper、TF64/TF_ON checkpoint、normalizer、原阈值、
K2/B2/bonus、prefix_then_cost、原 shared-prefix/O3、20/50 scheduler 与完整 reset bank。
唯一调度差异是 C 持续 S，P 以 S S T 重复；T 复用封存的原 canonical
target 路由，action50 次合法更新及其第51次终端 forward 全部计账。

固定六 pair 的身份、顺序、物理 GPU 及授权以独立 job 的 plan.json 为准。
两个 GPU 各一个持久 server，同 pair 两臂在同 GPU 顺序自然完成；
每轮两 pair 并行，波间检查新增成功损失与隔离失败。
首两 S 的现场原始状态、有效 KV、RNG、动作或初态不一致，被动保留，
完成已准入 pair 后停止新波；不将观察器失败改成机器人自然失败。
资源或路由硬限导致 PARTIAL，不伪造完整终态，也不自动重试。

## 入口

- `runner.py --job <absolute-job>`：已批准的完整面板；须先有部署及独立守卫收据。
- `server_entry.py` / `client_entry.py`：runner 管理的真实服务与完整 LIBERO 客户端。
- `controller.py` / `sealed_refresh.py`：成功 policy 请求计数与原 T 数学路径。
- `role_contract.py` / `evidence.py`：S/T 分开计账、实际调用/推进/完整区间。
- `state.py`：现场类型及原始字节比较，不计算摘要；完整 KV 只保留必要 RAM 引用。
- `deployment.py`：从本地封存的小 checkpoint/source 参照复用原预测与 bank 数学，
  不调用旧隐式 SHA loader。大 target 权重按原本地发布 revision 与大小归档，
  未进行远端文件 SHA，不宣称密码学身份复验。
- `report.py --job <absolute-job>`：只生成该面板的描述性报告，保留所有缺项与失败。
- `guard.py`：原绝对时钟，按 PID/出生时间/PGID 信号清理自有进程。
- `stage_remote.py` / `stage_parallel.py` / `download_assets.py`：真实迁移，不启动模型测试。
  SSH 密码只从终端读取并保留在内存；禁止将凭据写入参数、日志或结果。

不在 AutoDL 上执行 smoke、独立 CPU/GPU/CAL/fixture 测试或 SHA；
必要审计跟随实际完整任务，未执行的12-chunk独立门保持 UNKNOWN。
每局沿原 horizon800，实际796命令与失败800罚分分列。
主成本为完整 action 区间 `ΣT_round/Σg`，另列完整 policy 成本、
所有尝试成本和共同成功集合成本，不用 target 调用数量代替延迟。
本卡上的计时均为探索性资源/成本记录，不是本机 SXM4 正式性能结果。

旧 TF_ON NO-GO 与 engineering_quality_screen、independent_test_complete、
noninferiority_established、formal_performance_permission 四项 false 始终保留。
不自动开启训练、原缺项补跑、120、独立 TEST、正式性能或第二环境。
