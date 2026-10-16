# 6完整episode原生诊断：授权、启动与时间预估

记录：2026-10-11T08:02:48+08:00；仅投机采样，非tile。

用户明确回复“可以的，可以给我预估时间吗”，授权上一步建议6局，不仅是方案讨论。实际于2026-10-11T08:00:28.367271+08:00服务器独立tmux启动，当前running_diag_T；尚未完成，未声称找到物理根因。

## 固定范围与关卡

- T/S0/B_on×task0、init3/4，各臂两个独立EGL worker各1局，算法条件串行，一个A100；全部按800 simulator horizon合同完成，无人为chunk截断，policy步数排除5稳定化，失败如实保留。
- target/seed20260922/verifier/global阈值0.007839680526450805/scheduler/cache/bonus合同不变；T off、S0原B2、B_on夜间固定末步/rjs/cxz/论文探索/投机采样/results/runtime/p0-minimal-frozen-task0-20ep-20261011-004242/training/B_on.pt。teacher明确关闭，**不采teacher、不训练/RL/真实大小模型联训，不自动成本复测或追加episode**。
- 只启用既有native observer：双物体pose、同grip-site/夹爪qpos、左右指接触、各自In篮谓词、done/terminal RGB。特权字段仅诊断，不进入部署policy。接触不等于持稳，In是区域包含，不是篮子接触。
- fresh 16 CPU回归 passed；新增job运行包装6项scripted控制流/teacher-off合同检查，非真实模型数值验证。现有源码未改，仅新增job包装，复用已有并行worker实现并正确记录teacher关闭。
- fresh真实provider scripted on/off各16规定动作，状态/RNG读取不扰动、字段/goal合取检查PASS；0完整policy episode，单独前置检查，不算6局，不是完整WAM on/off数值证明。
- 启动前GPU0MiB/无可见compute进程，fresh source SHA与target9文件size/mtime一致，B_on content digest吻合；target非全权重content checksum。40GiB阶段门、12GiB紧急停，shared owned GPU lock，失败关卡停止，仅清理本任务自有进程。

## 时间估计（Asia/Shanghai）

对应旧6局每臂取最慢worker：T363.71、S0787.60、B_on714.97秒，相加约31.10分钟，再加3次模型加载/启动约35分钟。**运行预计35–50分钟；含前置检查与初步整理45–70分钟**；从08:00实际启动估计08:45–09:10初步结果，非承诺。诊断写盘、并行竞争、未复现旧失败或出现新长尾都可能改变时间，不为了时间目标截断。自动报告先保证数据完整，接触/模型根因可能需另一步，不能把“出报告”当作因果已证。运行plan的粗算字段先取了T init3而非两init较慢值，给出30.77分钟；此处按T init4较慢值更正为31.10分钟，原始plan不回写，不影响35–50分钟估计区间。

## 重要复现边界

本轮检查发现：LINGBOT_SEED只在server启动设置，_reset不重设，每chunk重新torch.randn。旧init3是worker0在init2后第二局，新init3是首局；旧前序chunk T17/S017/A_off17/B_on16。旧init5在init4后，前序T19/S050/A_off19/B_on19。本轮不恢复旧CUDA RNG或缓存/模拟器snapshot，**相同init/启动seed不保证旧失败精确复现**；历史数据仍是真实组合系统结果，但不当严格同噪声反事实。新3臂均按相同首局/worker合同比较。

自动报告比较native实际命令与旧已解码命令的共同前缀，并分别给出是否完全一致；有差异不能直接归因于observer。未复现旧失败时如实写未复现，不自动加局，不用旧视觉标签强行标新数据。完整WAM on/off数值等价仍未证明。

## 后台与自动产物

独立服务器tmux socket spec-native6-075638，session native6；supervisor PID 1796841，tmux daemon PPID1已核验，保留PTY stdin，子任务/dev/null。本机熄屏或SSH断开不终止它；服务器重启/硬件故障/OOM不作保证。

- [运行目录](../../../投机采样/results/runtime/p0-native-failure-task0-6ep-20261011-075638)、[状态](../../../投机采样/results/runtime/p0-native-failure-task0-6ep-20261011-075638/status.json)、[计划](../../../投机采样/results/runtime/p0-native-failure-task0-6ep-20261011-075638/plan.json)、[主日志](../../../review/OMISSIONS.md)。
- [HW自动结果入口](../../../投机采样/results/runtime/p0-native-failure-task0-6ep-20261011-075638/FINAL_REPORT.md)指向本任务FINAL_REPORT.md，完成或失败后自动写。完成后输出native_steps.csv、episode_diagnostics.csv、native-validation.json、diagnostic_metrics.json与原始native、terminal RGB、worker EGL审计。
- 停止：touch '/rjs/cxz/论文探索/投机采样/results/runtime/p0-native-failure-task0-6ep-20261011-075638/STOP_REQUESTED'。不直接关闭tmux绕过自有子进程清理。无自动重试，不覆盖夜间原数据。

**下一步：先读六局完成/失败关卡，CPU对齐接触/位姿/夹爪/各In与命令/图像，确认敏感阶段和返工解释是否成立；之后单独设计同负载T/A_off成本计数，不与这6局混跑。暂不上RL/KL或真实联训。**
