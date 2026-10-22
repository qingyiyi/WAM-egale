# 融合边界审查

- FFN1/GELU：BF16(sum+bias)先显式产生，再将该BF16转FP32执行原GELU公式，最后转BF16。debug写F1，普通路径仅写ACT。
- FFN2/残差3：第一K7168部分和先转BF16写F2；原grid同步后，第二部分和先加bias再加第一段BF16，转BF16后才做gate乘法和残差加法。最终F2仅debug保留，普通路径直接写OUT；部分和F2仍保留，未去掉整个F2 workspace。
- 两个融合只走candidate且M224分支。B2/action、B2/video仍走原阶段路径；GELU公式抽成同一inline helper，需覆盖B2有限核验。
- 保护原更新/可见KV/RNG；不修改scheduler、接受器或投机代码。当前不是ready调度，也没有新增专职warp。
- 源推导：warm独立GELU1568任务、残差336任务合入现有GEMM写回；短/中/长逻辑任务3400/3472/3400，grid同步29（原31）。这是源码口径，未插桩实测。
- 正常F1写+读减少12.25MiB；FFN2最终写与残差读F2减少2.625MiB，共14.875MiB逻辑全局张量访问。第一段F2写/读及ACT存取仍存在；不声称实际HBM字节同幅减少。
- 比较原Torch同缓存完整层；上一版自有只作为内部回归控制。精度门仍保留native共有中间量审计，native不作性能基线。
