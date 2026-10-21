# 当前整层剩余开销：三KV阶段与CTA时间线

状态：OBSERVED；本机NCU待空闲窗口。活动数学源码未修改，当前源 `df2a307241928567e2b0d80a5e6c601b8e1aedaf6e4713b693d88ce247ead11a`。

## 范围与数值

固定真实 B14 action layer0 short/middle/long fixture。420次完整调用、10962次阶段调用；三个fixture全部共享中间量、全部逻辑有效KV、输入、RNG有限逐位通过。不是所有输入、30层、机器人质量或正式性能证书。

插桩在当前函数调用点记录stage起止，并在每个CTA的阶段grid barrier前后记录globaltimer。原实现资源为108CTA、256线程、255寄存器/线程、98848B shared、1456B local；插桩local为1448B，其余相同。插桩包含额外清零kernel，原/插桩Graph交错中位数扰动：短1.093%、中1.018%、长0.997%。时间量化约1.024µs，不能解释亚微秒差。

native阶段是已对齐的native代理/库调用，不能将其中每个pointwise代理等同于Torch原始逐算子表达式。native阶段是执行同一native前序后的外侧device-marker间隔，包含marker/launch间隔；自有是整层函数调用点间隔，包含领取/计算/阶段同步。两种边界并不完全相同，未扣空marker。因此下表用于定位优先级，不能把差值逐项相加当成正式E2E归因，也不把原语时间与已有eager完整层表混算加速。

## 主要阶段：native / 自有，µs

|计算|短KV|中KV|长KV|
|---|---:|---:|---:|
|Cross K+V投影|1107.97 / 1244.16|1107.97 / 1243.14|1107.46 / 1244.16|
|六个小action投影|235.01 / 371.71|234.50 / 371.20|233.98 / 367.62|
|FFN1+FFN2|282.11 / 365.57|283.65 / 366.59|282.11 / 365.06|
|Cross K RMS|68.61 / 129.02|69.63 / 129.02|69.12 / 130.05|
|GELU|15.36 / 42.50|14.85 / 41.98|15.36 / 43.01|
|Self attention|111.62 / 134.14|376.83 / 352.26|821.76 / 601.09|
|Cross attention|99.33 / 143.36|99.33 / 143.36|99.33 / 143.36|

六个小投影是self Q/K/V/out及cross Q/out。完整逐阶段数据见STAGE-SUMMARY.csv；这些均已向量化/使用Tensor Core设备核心，不是把模型计算改成串行逐元素CPU式循环。

## 等待与计算要分开

`arrival_spread = 最后CTA到达 − 最早CTA到达`；`post_last_arrival = 最后CTA离开 − 最后CTA到达`。后者仍含插桩/释放传播，**不是纯barrier硬件成本**；前者常含其他CTA尚在做必要计算，**不能直接全部删除**。CTA等待可相互重叠，不可将108个CTA等待相加当整层延迟。

- 全部记录中post-last-arrival中位数1.024µs，最大4.096–5.120µs。没有证据支持“每个grid.sync本身耗时几十至上百微秒”。
- Cross K/V末尾到达跨度约45–46µs，平均每CTA等待约27–28µs；有尾部空闲，但整个阶段约620µs。仅最后屏障不是其全部差距。
- FFN1最后16个M64任务的尾段跨度32.768µs，平均CTA等待约29.1µs。92个CTA没有这批tail任务；这是已测的补位机会，不等于已实现收益。
- FFN2第一道内部barrier到达跨度约116µs；108中仅96个CTA做GEMM，未分配任务的12个CTA先到等待。平均CTA等待约18µs，不能把116µs解释为全部CTA的同步损失。
- Self attention尾部到达跨度随KV从20.48增至65.54、120.83µs。长KV原语本身反而快于native；不能因此说长KV无瓶颈，也不能把其原语收益当整层收益。
- Cross K RMS末尾跨度仅约4.10µs，GELU约3.07µs，而各自计算段约129–130µs、42–43µs。它们的主要差距不在最后几微秒的grid barrier；需检查阶段内部队列/CTA同步与资源限制。

## 源码对应与待证机制

`whole_layer.cu` 的begin_stage/claim_tile/finish_tile仍分别初始化阶段计数、atomic领取、CTA同步；没有跨阶段ready调度。Cross K RMS每个tile两行，7168行对应3584任务，每任务一次队列领取和多次CTA/128-thread组同步。GELU处理224×14336个元素，每2048元素一任务，共1568任务；虽为vec4向量读写，调度粒度仍可能偏细。此处是源码支持的机制假设，需要受控调整或硬件计数才能量化。

CUTLASS主循环只由warp3–6的4个warp执行，8个warp参加写回；整层固定资源限制每SM一个CTA。静态255regs/1456B local不能证明动态spill流量。NCU将核对实际Tensor管线活跃度、local load/store、L2/DRAM、barrier/scoreboard及eligible warps；权限或资源不足则保持UNKNOWN。NCU的barrier-stall不是grid.sync专用计数，不能直接称为全局同步占比；阶段末尾等待仍以CTA时间线解释。三对独立GEMM采用hot warmup，完整层具有真实前序，二者缓存条件不同，不用hot计数冒充整层每阶段流量。

下一步优先原语：同条件定位Cross K/V和小投影的吞吐差；Cross K RMS/GELU降低每tile的队列/同步开销；最后才把已测尾部空位交给无依赖Cross K/V。不能只删除全局同步，否则破坏就绪与shared workspace语义。

原始证据：trace-r2/TRACE-RESULT.json、CTA-*.csv、TRACE-STATUS.json；汇总STAGE-SUMMARY.csv、BARRIER-SUMMARY.csv、TIMELINE-SUMMARY.json。分析脚本analyze_timeline.py。未修改baseline、optimized数学、投机生产或四权限false。

## 既有受控对照提供的补充（非本轮新测）

此前pointwise-isolation真实Torch GELU为vec4、128线程、3136CTA，约10.4–11.2µs；自有全网格vec4约10.9–11.8µs、有限逐位通过。固定108CTA标量即使没有统一队列/shared预留也约43.8–44.9µs。它同时改变布局和向量化，不能单独量化当前vec4的队列开销，但支持“小数学体能快，统一执行资源不自动继承快”的判断。当前GELU的native代理约15µs，比真实Torch表达式更慢，不能据此说当前GELU已接近Torch。

此前minor-ops的Cross K RMS适配独立约72.2µs，带统一执行资源的阶段约121.9µs；本轮嵌入结果129–130µs。旧记录不是新版本同源独立对照，不跨版本相减推断spill。链接：../pointwise-isolation-20261021/RESULT.md、../minor-ops-20261021/RESULT.md。
