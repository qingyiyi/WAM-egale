# M37：设备数值契约有限检验与Long/Object成本归因

登记时间：2026-10-20T09:24:01.490859+08:00。用户明确同意继续上次下一步。不是旧M34/M35续余额，不重跑旧任务。

- [x] C1：用现有330局完整时间账本解释Long/Object；所有失败与共同成功条件口径同时保留。
- [x] C2：保存本轮源/输入身份、保护项；在远端空卡读取公共cuBLASLt heuristic。它是描述符重建，不是拦截原调用，不是测得occupancy。
- [x] C3：唯一设备候选：CUTLASS SM80、按实际Q/FFN2各自CTA/warp/stage与splitK配置；使用库完整mainloop/epilogue/semaphore，不重复旧四种只改变workspace累加公式。宏几何一致仍不保证逐位。
- [x] C4：两个既有short真实action输入（B2/B14）同输入Q和FFN2与原Torch比；保留 BF16输出与bias语义。最多120 primitive调用、8完整block、0full30/robot episode。数值失败即停止整层接入/后续性能，不用未准入原型作正结果。
- [x] C5：归档结论与清理；只有上述相关数值门成立才进入异步tile整层集成，否则提出明确缺口，不盲扫。

CPU源码/编译最多2核、约2小时；新GPU生命周期45分钟（含准备/等待/失败/清理），只用远端GPU0一worker；启动时写新T0、35分钟工作截止及45分钟清理截止，失败/超时不自动重试。新存储≤512MiB，不复制权重/完整KV，不安装依赖。仅使用已有SM80 CUDA12.4/Torch2.9.0+cu126/固定CUTLASS。明确的编译错误可保留原失败后在既定CPU预算内修正，不把失败静默覆盖。

输入是M34/M32原真实fixture，不采新数据。只修改optimized独立设备组件与本namespace脚本，baseline/投机/原tile/HW保护文件不动。强cached/shared-prefix/Graph路径是后继baseline；本次数值门失败不进入计时。所有四质量字段false、默认A_off、历史NO_GO保持。进程以PID+出生时间核对再仅清理owned；NVML host PID与容器PID分开。

## 首次入口修正

metadata-1为0完整block/0heuristic的DSO路径实施失败，原记录保留。仅独立新脚本换到远端已部署的冻结sequential调试入口；数学/依赖不修改，Q/FFN2仍直接比原Torch。entry-corrected-1沿用原T0与硬截止，消耗合并；不重试算法失败或完整episode。详见ENTRY-IMPLEMENTATION-ISSUE.md。

实际终态：四原语NO-GO，数值门失败，异步整层集成未触发；有界工作完成，不代表科学准入。见RESULT.md。
