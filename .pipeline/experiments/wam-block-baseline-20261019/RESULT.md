# 完整block顺序基线：CPU阶段结果

生成：2026-10-19T02:11:57+08:00。固定base `ca5f37e0482c2e38e3738e3d8e71e3b185990a23`；状态 **CPU_BASELINE_ESTABLISHED_GPU_PENDING**。

A. 以前只有SDPA独立替换、原block混合执行；现在新增可单独调用、显式阶段的B0_HYBRID_SEQUENTIAL。它仍依赖Torch GEMM/norm/激活，不是全自有CUDA。

B. 六个真实action block0输入与六个同输入合法写控制，36次CPU完整执行通过R0/R0、R0/B0逻辑字节、有效KV有限及metadata/输入/RNG检查；两真实fixture各28个观察阶段、静态投影控制通过。合计仍是六个唯一真实fixture，不是独立机器人样本。

C. 原保存过程已连续化，原CUDA stride/alias/RNG缺失保持UNKNOWN。CPU与历史GPU gold不逐位，完整差异保留；B0 GPU/block以外范围均待验。四项机器人质量/正式性能权限保持false。

D. 新B0未测GPU性能。旧R1的54个原版block三轮中45点波动>5%、最大7.387倍，不能据少轮中位数保证稳定对齐。新计时先A/A再ABBA/BAAB；54行性能表都是待运行，不预填0ms/PASS。

E. 下一步先补六fixture的GPU数值/状态与可比较完整block时间，随后才依据profile选一个D0；当前不开发tile scheduler。预算与权限见[BUDGET-AND-NEXT-STAGE](BUDGET-AND-NEXT-STAGE.md)。

证据导航：[事实/未知](FACTS-AND-UNKNOWNS.md)、[覆盖](BASELINE-COVERAGE.md)、[契约/DAG](BLOCK-CONTRACT.md)、[数值字段](ACCURACY-RESULT.json)、[性能限制](PERFORMANCE-LIMITATIONS.md)、[readiness](SCHEDULER-READINESS.md)、[claim](CLAIM-EVIDENCE-MATRIX.md)、[资源](RESOURCE-AND-STOP-REPORT.md)、[源码来源](SOURCE-EVIDENCE-PACK.md)。
