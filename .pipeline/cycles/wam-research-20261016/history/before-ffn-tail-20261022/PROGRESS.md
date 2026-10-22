---
kind: progress
cycle: wam-research-20261016
plan: PLAN.md
status: active
updated: '2026-10-22T18:59:32.671003+08:00'
current: M39
next: FFN诊断完成；先汇报尾部生产/写回/函数保存恢复证据，再决定单因素优化，不自动实施。
---

# 当前研究进度

M39 FFN诊断完成；当前R2源码/DSO未改。真实短/中/长B14独立FFN（含GELU、门控残差）相对原Torch为0.8080/0.8095/0.8102×，慢23.4%–23.8%。两进程三KV有限输出/输入/KV/RNG通过；嵌入整层与独立FFN插桩span差不到1%，四科学权限false。

72条时间线均显示最后4个FFN1 tile拖后组3；末尾FFN2计算约68%、写回约25%、提交等待不足约1%。本机NCU诊断同计算路径local读27.54MB/写28.15MB，原Torch为0；生产SASS主要为非内联函数入口/出口寄存器保存恢复，检查的MMA区间无LDL/STL。local请求不是HBM字节，具体延迟贡献尚未量化。原Torch FFN1同样112CTA，不能只按tile数解释慢因。

[诊断结果](../../experiments/single-layer-async-gemm-20261020/ffn-four-group-pipeline-20261022/diagnosis/RESULT.md)；[时间线](../../experiments/single-layer-async-gemm-20261020/ffn-four-group-pipeline-20261022/diagnosis/FFN-TIMELINE.png)。两端自有任务已退出；本机后来出现hlj任务，未干预。GPU worker98.889秒，保守含准备/清理1039.686秒，累计10884.364/36000秒；本次19:24:12硬截止与旧截止保留。PID命名空间误判、捕获未replay比较、远端NCU权限失败均保留；同DSO修正脚本后完成，本机NCU为计数补采、不混计时。

此前四组R2相对原Torch完整单层1.1646/1.6808/2.7386×，较同轮此前阶段版仅改善0.33%–0.36%；不是本轮新整层Torch比值。数学tile3400/3472/3400、27grid；ACT仍落全局，只有FFN局部流水。baseline/投机/原tile/保护文件不改。

| ID | 阶段 | 状态 | 当前结果 / 证据 | 下一步 |
| --- | --- | --- | --- | --- |
| `M39` | 单层优化 | in_progress | FFN瓶颈定位完成，有限正确性通过 | 优先FFN1尾部工作分摊；随后FFN2写回与函数保存恢复，尚未执行 |
| `M17` | LIBERO证据 | blocked | 四套6000局齐；非独立TEST，首两S/暴露/长尾缺口保留 | 不恢复过期作业 |
| `M40` | RoboTwin | blocked | 原DEV90完成；原截止过期，1800局/160h未获准 | 扩大资源独立处理 |

[PLAN](PLAN.md)；[此前状态](history/before-ffn-diagnosis-20261022/PROGRESS.md)。10/22 go/no-go、10/23身份冻结、10/28主表、11/01 18:00中文证据包；证据不足收缩claim。
