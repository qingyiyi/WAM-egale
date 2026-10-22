# 当前单层算子带读入口

请用中文带我读当前optimized源码，先读WHOLE_LAYER.md和STAGE-MAP.md，明确现在是一层一个persistent kernel、108CTA、阶段grid同步，不是30层大算子。

从whole_layer.py的prepare_context(context, precompute=True)和__call__开始：条件按字节分组、GPU预计算K0/K/V、内容变化须重新prepare并重录Graph。沿csrc/whole_layer.cu的forward与whole_layer依次读Self QKV、RMS+RoPE、Self attention、Cross Q/已缓存K/V、FFN及残差；attention设备函数在csrc/fa2_stage.cuh，CUTLASS投影在csrc/cutlass_gemm.cuh。

解释每阶段公式、tile、warp职责、完整行归约和grid同步。warm5304/5376tile、31grid；uncached6200/6272tile、36grid。候选私有KV不共享，Cross条件映射与Self历史映射不同。RMS没有与GEMM tile融合。

完整阶段耗时见STAGE-MAP；正常warm短中长0.957/1.124/1.140ms，对同缓存guarded仍慢7.94%–11.70%。逐阶段来自独立profile，约1.6%–2.0%扰动，不能当正常时间或把CTA平均等待相加当墙钟。首诊断漏记两处同步的失败已保留并修复，不要据旧输出改写质量门。

baseline、投机侧、原tile不可修改；本次只讲解，不自行改代码。有限单层逐位验证不代表全部模型/机器人输入证明。
