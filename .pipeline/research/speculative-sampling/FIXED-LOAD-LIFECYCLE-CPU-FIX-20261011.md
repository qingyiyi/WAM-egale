# 固定负载启动生命周期修复与 CPU 回归

日期：2026-10-11T20:47:48.550829+08:00（Asia/Shanghai）。归属：投机采样，非 tile。

**本关 CPU PASS；新 GPU 尚未启动，性能/恢复/不扰动未通过。** 用户已确认原 CPU-first 方案并继续到正常启动及 GPU 测试有结果；沿用新 job 0完整episode/2EGL/168固定chunk/90分钟上限，关失败停、不补跑、不训练。

## 根因与假设验证

1. **reset 前缺字段（主因）**：原 constructor 不建立 action_per_frame 等推理状态，原 reset 才建立。旧冻结 worker 在新鲜、不预填几何的回归中复现 AttributeError；新代码通过。
2. **off-mode guard（已修的旧问题）**：保持原公共 guard；只在原 reset 后临时 k2b2，load/to/eval 全部异常恢复原 mode。4 原模式及 7 生命周期故障回归通过。
3. **ready/重复执行边界（已验证）**：bootstrap ready 只表示 target 就绪、draft 尚未加载；另有原 reset 后 draft-ready。任意 build 尝试都消耗一次机会，成功进入 prefix 或失败后均拒绝第二次 build；不制造假 fixture/draft ready。
4. **prefix/RNG/真实 GPU 完整性（仍待 GPU）**：预加载位于初态编码、记录 prefix、fixture seed 和计时之前；未改原 prefix/cache/去噪/scheduler/阈值/bonus。CPU 未验证真实大模型/cache/编码，必须用后续 GPU 恢复与 G1。

## 变更与验证

- 仅任务 `worker.py` 和 `test_cpu.py` 两个实现文件。Runtime 只有 constructor/build 的 AST 改变；其余 Runtime 方法保持。公共 server/client 未改。
- 固定负载 **89 passed**，合并 **221 passed**；新增12项生命周期顺序/4模式/7故障/旧码复现。保存实际命令、退出码0和原日志。
- 原 va 环境直接调用 **真实 VA_Server.__init__ → Runtime bootstrap → 原 infer/reset → 真实 A_off 加载 → 原初态 dispatch → 第一 prefix 边界**；没有手动补 reset-owned 字段。A_off 输出112维/eval/CPU权重逐项一致，off恢复，重复build拒绝。
- CPU 重组件替身是显式的：target/VAE/text/tokenizer 加载、cache创建、FSDP配置、prompt/obs编码。原构造后仅将执行设备/dtype转为CPU上下文；不合成推理字段。探针在第一真实 target forward 前主动停止，不能冒充GPU整体启动或成本结果。
- 1271个基线文件中只有获准两实现改变，其余1269个SHA不变；11旧实现原SHA归档；9target文件size/mtime不变。旧两失败job/plan/gate/报告保留，不恢复或覆盖。target元数据检查不是全权重哈希。
- self执行：同一agent修改/运行测试，未启动subagent；另在不同Python进程直接导入原模块做实际构造probe，减轻但不消除同一作者验证的局限。

## CPU 与 GPU 成本职责澄清

性能比较始终计划在 A100 上真实执行：T是原target，A_off是draft+冻结target验证的投机路径。**CPU预检不是CPU模型测速；CPU比较器读取GPU计时/输出做一致性与统计。** GPU worker计时前后CUDA同步，CPU时钟测的是GPU路径实际wall time，包含必要主机调度，不等于仅kernel时间。额外输出复制/hash/存盘在测量窗口之后。之前“CPU整理成本账本”是分析既有GPU记录，不是改用CPU推理。此前两次GPU都止于启动，没有本轮四格成本测量。

## 下一步

新建GPU job；冻结修复后的实现和原输入，重新实际CPU预检与原模块probe，通过后独占/空闲资源门→两模型bootstrap→两独立EGL初态→原reset/draft/prefix与全30block/cache/RNG恢复→不扰动G1→短/长KV×单/双需求GPU成本。每关HW记录，失败即停，不补局/调用、不训练。不能原样恢复旧one-shot或拿CPU PASS充当GPU结果。

证据：
- [CPU修复job](../../../投机采样/results/runtime/p0-fixed-load-lifecycle-cpu-fix-20261011-203648)
- [221项日志](../../../review/OMISSIONS.md)；[实际退出码](../../../投机采样/results/runtime/p0-fixed-load-lifecycle-cpu-fix-20261011-203648/combined-CPU-process.json)
- [原模块构造/生命周期CPU probe](../../../投机采样/results/runtime/p0-fixed-load-lifecycle-cpu-fix-20261011-203648/real-constructor-CPU-result.json)；[probe源文件](../../../review/OMISSIONS.md)
- [修复完整性验证](../../../投机采样/results/runtime/p0-fixed-load-lifecycle-cpu-fix-20261011-203648/CPU-LIFECYCLE-REPAIR-VALIDATION.json)；[worker diff](../../../review/OMISSIONS.md)
- [第二次GPU失败历史](FIXED-LOAD-COST-ATTEMPT-20261011-200434.md)
