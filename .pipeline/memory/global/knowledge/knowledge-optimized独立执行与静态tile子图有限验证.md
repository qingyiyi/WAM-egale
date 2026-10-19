---
kind: knowledge
name: optimized独立执行与静态tile子图有限验证
level: reference
scope: cycle:wam-research-20261016;M31
status: active
updated: '2026-10-19T19:27:26.713126+08:00'
sources:
  - .pipeline/experiments/transformer-tile-subgraph-20261019/RESULT.md
---
# M31执行路径与证据范围

optimized已不继承baseline，独立参数/native/C++与wam_optimized_block_cuda；旧baseline的88项源/依赖/保护文件不变。默认sequential，tiled和FFN1融合独立开关。单独进程9真实输入/54状态通过，无baseline模块/旧WAM动态库加载。

self/cross/FFN实际子图按同stream阶段执行；attention显式query tile与FFN1显式row/column MMA设备函数已接入。QKV/Wo/FFN2仍cuBLASLt，未实现persistent/ready队列。旧FFN2数值失败保留。

最终六action（三KV×B2/B14）及三video Q128，共9block；72KV状态、输入/RNG/参数地址与grad flags/API安装还原逐位通过。kernel名审计确认实际tile路径。首次工具误判/标签/收据问题保留，不改旧失败。

新full30受GPU显存阻塞，未做新性能或机器人任务；旧M28完整模型证据不能继承到新实现。此次是可执行tile子图基础，不是已证实MegaTransformer加速。
