---
kind: experiment
id: draft-teacher-path-20261016
cycles: [wam-research-20261016]
status: completed_bounded_diagnostic
---

# draft 收口：真实 teacher 路径诊断

**本阶段已完成并清理**：[结果](RESULT.md)；[以TF为唯一主模型的下一步](NEXT.md)。两完整记录policy chunk、同query及时间条件反事实完成；不是新episode。TF64主模型已明确，当前权重仍NO-GO、训练机制尚未触发，旧四权限false。

用户本阶段优先完成 draft，见[决定](../../memory/global/requirements/decision-当前阶段优先完成draft收口.md)。本实验只排查 draft 质量所依赖的 target/verifier 差异，不开发新算子，不训练，不启动旧120。

固定公开基线 `b0dfac3a02f65de33a1217246b83003615166640`。新入口为 `投机采样/experiments/draft_teacher_path_diagnostic_v1/`，原 source、tile、FAILED、retention 与四项 false 权限不变。

使用已解盲 task4/init26 的 T/TF repeat0 初始相机、prompt、实际 target RNG 和完整首 policy chunk。先重现两臂 actions/video、全部50合法 scheduler 提交及物理动作；从 T 的第二次更新和 TF 的第一棵候选树捕获真实模型输入与30层完整有效KV。全部输入、历史和RNG相同才准入同query归因。

对这个冻结真实 query 比较 B2普通、B2只读、B14原生、B14已有shared-prefix。记录输入条件、首层各子算子及全30层输出、target direction 和实际 query 重复性。不将一个7候选批次改成Q112，不合并私有KV。

预算：CPU准备≤2核/60分钟；单本机GPU worker、CPU12–13各2线程；GPU作业墙钟20分钟（含等待/setup/失败），17分钟停工、3分钟清理，实际准入还受登记绝对截止约束。新增存储≤1GiB、磁盘保留5GiB、显存准入22GiB。共享质量允许，计时不进性能表。实际启动前独立 `PREREGISTRATION.json` 固定源、输入和预算，外部OS timeout与CPU监督只清理自有进程；不自动重试或延长。

这是两个完整的已记录 policy chunk 正确性重放，**不是两个完整机器人episode**，不补齐M11缺失第二TF的质量任务，不产生独立TEST，也不能解释整局失败的因果贡献。训练只有具体递归分布/监督目标机制证据支持时才另登记，当前仍HOLD。

Attempt r1已事前登记：[PREREGISTRATION](../../../投机采样/results/runtime/p0-draft-teacher-path-diagnostic-v1-20261016-r1/PREREGISTRATION.json)。原190冻结源及新增入口/共享工具校验通过；worker与supervisor真实依赖在各自独立CPU进程导入通过，CUDA未初始化。合并到同一CPU检查进程会触发旧模块cleanup重名保护；实际架构已隔离，不为此修改旧源。

终态与原始证据待实际作业填写；失败或超时亦保留，不把准备完成写成执行完成。

## r1终态及唯一局部因果追加

r1已完成：[实际数值](../../../投机采样/results/runtime/p0-draft-teacher-path-diagnostic-v1-20261016-r1/WORKER-RESULT.json)。两完整记录policy chunk、各50合法提交和视频/actions/物理动作逐位重现；同query全部原始字段/30层有效KV/实际RNG相同。B2普通=B2只读；B14原生=已有shared-prefix；首个被观察到的数值分歧在`condition_embedder_action`，早于第一个block。此证据没有解释整局失败，不触发训练。

r1实际监督墙钟211.860秒、117 target forward、443399880B，自有worker及metadata进程均退出，未用信号。为缩小这一已定位模块内部原因，新增独立r2事前登记：仅一个真实query，模块子算子观察及B2时间编码替换的两项反事实，最多8次完整verifier；不训练、不运行episode、不集成新路径。

**r2仍从r1实际开始累计20分钟**，含失败/准备间隔，沿原17分钟停工/3分钟清理，不延长旧窗；若登记前剩余时间不足则不启动。代码只在`draft_timestep_causal_v1`，CPU监督复用r1冻结字节码，仅绑定新worker目录，原模型/旧入口不改。实际登记/终态随后补指针；观察结果不能当新生产策略或质量证书。

实际启动：2026-10-16T19:28:21.985397+08:00，CPU supervisor PID3571999，GPU worker身份见runtime的OWNED文件；最大20分钟，未注册其他GPU任务。
