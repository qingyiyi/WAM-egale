# 两份算子实现整理与实际30层验证

更新时间：2026-10-19T11:03:23.013557+08:00

代码已集中到`算子研究/wam_ops/baseline`、`optimized`，共享安装与验证/计时工具。已删除16个无投机外部调用的旧算子实验目录；308源留Git a0ed2d8e49403cdca8d87afd40b21dd19595710c。26444694字节旧目录与5044712字节重复build已删除；这只是代码/构建清理，不冒称解决整体数据盘占用。

1875份投机源/配置/依赖清理前后未改。原tile仍被shared-prefix构建脚本引用，原位只读保留。所有原始实验结果和失败保留。

## 当前验证结果

新baseline完整block前向与固定ca5f37e逐AST一致，删除了未使用的其他cache引擎分支；参数名、dtype和舍入边界保留。optimized复用同一block，仅安装独立CUDA SDPA；CUDA数学源与旧已对齐v2逐字相同，改扩展名称并按同编译选项重新构建。原模型源码与固定ca5f37e逐字相同，不修改投机服务源码。

完整一次六点矩阵已经执行：真实B2/B14 block0×短/中/长KV，实际30层B14与同root派生B2×短/中/长，以及short B2的合法write1/2。原版/baseline/optimized共54项数值/有效状态/输入/RNG检查全部通过、max_abs=0；B14完整模型历史GPU gold也通过。较早首轮短KV22项通过单独保留，不当成额外独立输入。实际安装/还原与所有参数对象/地址也通过。

B2 full30使用真实verifier根的CFG0/7行并走普通B2临时cache路径，明确为派生诊断，非保存的原始顺序B2轨迹。模型为实际30层未分片实例，不冒称额外FSDP路径验证。原生产CUDA stride/alias/RNG缺失仍保留；完整机器人policy/接受路径未由本项新增验证。

## 计时状态

完整六点block/full30三臂、三轮交错数据已经采到。但结束资源检查发现外来GPU进程，该轮原状态FAILED/PARTIAL保留，数据只作探索性，不签正式性能。首轮短KV也因资源中断保留。剩余54次full30预算登记为continuation-2的最后一次original/baseline对照，等待用户要求的自然空闲，不再要求协调窗口；原起点/13:14截止和150次full30帽不变。

原始数值与计时见[完整矩阵](continuation-1/process-1/RESULT.json)，[首次短KV](process-1/RESULT.json)，[最后接续登记](continuation-2/PLAN.md)。结束后更新本页实际终态，不回写旧attempt。

代码阅读见[算子README](../../../算子研究/README.md)。四项质量/正式性能字段仍false，算子有限验证不放行TF正式45chunk/独立TEST。
