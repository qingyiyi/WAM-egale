# 受控同init T/S0原生对照：启动记录

更新：2026-10-11T13:18:44+08:00（Asia/Shanghai）。用户确认前一轮建议的4局上限；结果尚未完成，不将启动写成验证成功。

## 本关探索

前一轮S0稳定init4 600步中，第二罐建立抓取前四闭合段无采样双指contact、抬升≤6.55mm；init3为293步。但不同init不能说明K2相对target的额外返工。本关统一seed/首局/原生字段，比较同init T/S0的第二次抓取与第一分叉。

## 执行与停止合同

1. S0先运行init3/4两个独立EGL worker完整首局（最多2局）。
2. fresh S0必须复现旧ON 293/600步与19/38chunk，且20初始状态数组、逐step state/RNG/RGB/command/done/reward、原生物理字段、全部完整30通道action与video张量相同。仅步数/成功相同不足。CPU关卡失败即停止，T不启动，不补局重试。
3. 通过后T mode off同样两个worker首局init3/4（另2局）。完整horizon800/5稳定步，无policy chunk截断，最多4局。
4. CPU保存完整对照/第一分叉/第二罐闭合段/阶段指标和结果。In是区域包含，不等于释放落稳；双指contact不等于稳定抓取。

## 已完成准备

13项fresh CPU回归、17项job-only包装合同通过；比较器旧ON自比较895行/114完整action-video张量全相同，故障注入step1状态hash正确拒绝。冻结240输入hash+9 target size/mtime合同，公共源码/旧结果/protected不改。

冻结target、S0 checkpoint、阈值、bonus关闭、cache/attention/scheduler/CFG、部署动作前缀与执行stride。native与公共指纹两组都开，teacher/trace/profile/训练/RL/联训关；不是clean timing。

## 实际启动

2026-10-11T13:16:48.534856+08:00服务器tmux后台启动；socket spec-paired4-20261011-131403，session paired4；daemon PPID1，supervisor 2346728，stdin PTY，child /dev/null。已看到两个独立server实际argv并加载，当前0/4完成，EGL live audit将在client启动后保存。GPU启动前0MiB/compute空，磁盘62.03GiB；共享独占锁，条件串行、不争用tile。端口19651/19652/29651/29652，预计运行20–35分钟（非保证），CPU/HW整理另需时间。

本地熄屏或SSH断连不终止该服务器任务；不保证服务器重启/停机/GPU故障仍存活。自动关卡和结果报告不依赖当前聊天，只有自身pid组清理。

运行：[status](../../../投机采样/results/runtime/p0-controlled-target-s0-task0-4ep-20261011-131403/status.json)；[plan](../../../投机采样/results/runtime/p0-controlled-target-s0-task0-4ep-20261011-131403/plan.json)；[自动结果](../../../投机采样/results/runtime/p0-controlled-target-s0-task0-4ep-20261011-131403/FINAL_REPORT.md)（完成/失败后生成）；[旧ON参考](../../../投机采样/results/runtime/p0-repro-observer-task0-6ep-20261011-120724)。

**下一步**：等待S0全测复现→T两局→CPU同init第二次抓取/返工定位；然后另行决定成本T/A_off同步包装/诊断和困难阶段同query监督。本轮不启动成本测试或新训练，不声称正式非劣、自然E2E加速或L2/z唯一根因。
