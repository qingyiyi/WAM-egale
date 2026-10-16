# 完整 episode 投机 E2E 结果与下一步

2026-10-13T09:38:15+08:00 | HW chat / research | zh-CN | Asia/Shanghai

**本次两个受控初态的完整任务，S0 投机相对原 target 加速 1.0817×，总耗时下降 7.55%，属于正向 pilot。** 相对同一冻结 draft 的原投机版 D，共享额外加速 1.0931×。环境轨迹 283/291 步，原 target 293/300 步，合计缩短 3.20%；全部六个 episode 成功。两初态各只有一个样本，不是多任务/多随机种子的稳定收益声明。

## 完整任务结果

T=原 target；D=原 K2B2 投机；S=D 加已对齐原 Torch 的 self-attention 历史共享 S0。使用当前冻结 A_off.pt、global selector、阈值0.007839680526450805。prompt/cross-KV、行证明复用关闭；scheduler/50 action更新/CFG/完整800 horizon 保持原策略。

| 初态 | T秒 | D秒 | S秒 | T/S | D/S | 环境步数T/D/S | 任务结果 |
|---|---:|---:|---:|---:|---:|---|---|
| 3 | 360.235 | 360.407 | 329.583 | 1.0930× | 1.0935× | 293 / 283 / 283 | 全成功 |
| 4 | 360.579 | 367.993 | 336.778 | 1.0707× | 1.0927× | 300 / 291 / 291 | 全成功 |
| 合计 | 720.813 | 728.401 | 666.360 | 1.0817× | 1.0931× | 593 / 574 / 574 | 6/6成功 |

原投机 D 相对 T 合计 0.9896×，本次接近持平、略慢；加 S0 后才取得完整任务的正向收益。速度比是基线耗时/候选耗时，1.0817× 不等于耗时下降8.17%，实际下降7.55%。

主口径为原客户端 elapsed_sec：包含环境 reset/稳定、prompt、所有 draft/验证/回退、动作提交、环境步进和 KV 更新，成功或完整 horizon 才结束。EGL 环境构造/close 的外层 wall 也记录，T/S 合计为 1.0820×，结论一致。模型 bootstrap 和计时结束后的文件 drain 不计入任务；没有按 target 轨迹长度提前截断。

## 时间边界与收益解释

每条件两个独立 EGL worker 同时运行；不同条件分批。profile、teacher、完整 server trace、逐步重型状态/RGB诊断关闭；共同计数在 RAM，无逐前向文件 I/O/CUDA event。实际命令只存 RAM，初末状态和命令在任务计时结束后保存。原模型已有异步输出保存保留在所有条件中。

reset/T5 为共同且不在 S0 中的阶段，T/D/S 每 episode 平均为 116.558/115.097/110.918 秒；跨批约4–6秒的差异不能归因于共享。独立扣除 reset 后，T/S仍为 1.0971×、D/S为 1.1208×；只比较完整 policy 调用，D/S为 1.1317×。因此本次净收益不依赖 reset 变快，但单次两初态与按批顺序的测量仍需扩大样本验证。

init3/4 的 S 提交900/950次 action scheduler更新，其中674/706次为draft；接受深度2分别303/501、312/525棵树。共享对应1026次真实七候选验证。D/S model前向分别959/1010，T各1404；同一初态的D/S调用分布、接受深度和提交数完全一致。

## 精度与轨迹核对

先做真实固定短 KV 完整 chunk D/S GPU probe，共102次 target前向。完整30通道归一化动作、视频、返回动作、30层有效KV及元数据、scheduler、streaming VAE和CPU/CUDA/Python/NumPy RNG均逐位一致。

完整任务中，独立 CPU 重载74对动作/视频 tensor、927,072个元素，逐字节一致；还核对所有保存的观察输入。574条实际执行命令、初末 simulator/controller状态和客户端RNG一致。所有条件同初态的初始状态、初始观察、XML和RNG相同。完整任务没有保存每个时刻的server KV或随机快照，不将短KV probe的缓存检查描述为所有chunk都做过缓存快照检查。

原投机 D 与原 target T 的初始视频相同，但首chunk动作有近似差异；这轮不能把 draft 近似与batch14/batch2数值形状差异分别定量归因。S与D全任务一致，说明这次共享没有额外改变轨迹。具体误差与控制方案见 /rjs/cxz/论文探索/.pipeline/research/speculative-sampling/TRACE-LENGTH-CONTROL-PLAN-20261013.md。

## 编排、旧证据与预算

T两任务在v1完整完成并冻结。进入D时，启动器读取新进程身份失败；之后观察到完整argv与任务匹配，推断为启动窗口的身份竞争。原始错误未保存当时argv，不能声称已经证明当时一定为空。新启动器先登记Popen、最多2秒等待稳定身份、失败清理刚创建且身份未变的PID；CPU验证瞬时空argv、PID变化拒绝、失败清理和5次真实启动。遗漏的本任务客户端按完整argv、任务环境、UID、PGID与start_ticks精确清理；原FAILED.json保留，清理间隔保守计入时间，两个未闭合D预留各4000全额收费。

v2在40GiB初始磁盘门停止，GPU/模型/前向都为0。v3按四新增episode设置1GiB输出硬限、16GiB启动余量与12GiB运行余量，保留所有旧证据；CPU22项通过。T按唯一初态复用已完成数据，D/S在v3补齐；模型、客户端、计时和共享代码与v1逐字节相同。所有失败与跨job复用均保留可追溯证据，未删慢样本或重写旧结果。

最新预算累计 21489 target intents /35000（包含未闭合预留的保守收费）、GPU 6545.956秒，即 1.818小时 /5小时。GPU最终显存0、compute/graphics进程为空、所有本次子进程已退出。生产代码和受保护 state/rules/cycle保持原状，私有原型默认关闭。

## 下一步

当前两个初态已经满足“轨迹至少与原来差不多”。下一步以S0和现阈值为基线，在长尾初态研究**每chunk累计加权误差预算＋夹爪/动作通道约束；预算用尽后剩余denoise走原batch2 target**。完整环境horizon和成功判定不变，以成功、环境步数和E2E联合验收。控制方案属于下一阶段设计，本轮没有启用新策略。

证据：/rjs/cxz/论文探索/投机采样/results/runtime/p0-speculative-e2e-v3-20261013-090832/FINAL_RESULT.json；独立审计 /rjs/cxz/论文探索/.pipeline/research/speculative-sampling/evidence/speculative-e2e-20261013/INDEPENDENT-E2E-AUDIT.json；CPU /rjs/cxz/论文探索/投机采样/results/runtime/p0-speculative-e2e-v3-20261013-090832/preflight/CPU-GATE.json；GPU probe /rjs/cxz/论文探索/投机采样/results/runtime/p0-speculative-e2e-v1-20261013-084008/probe/PROBE-RESULT.json。
