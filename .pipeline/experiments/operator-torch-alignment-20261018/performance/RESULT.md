# 新v2性能复测：attention接近Torch，完整block仍有尾部

新v2与Torch在本次实际输入下可直接比较。attention Graph-core中位差约0.14%/0.19%，与噪声相近；完整block存在9/54超过5%的慢点，完整30层六query中两例超过5%。不能宣称所有形状稳定不退化，也不把更快点当新系统创新。

| 层级 | 点数 | 新/Torch中位数 | 最小–最大 | 慢超过5% |
|---|---:|---:|---:|---:|
| self | 54 | 1.001441 | 0.963025–1.045973 | 0 |
| cross | 54 | 1.001933 | 0.996350–1.039394 | 0 |
| block | 54 | 0.984655 | 0.410135–1.343666 | 9 |
| 完整30层B14 | 6 | 0.993934 | 0.739267–1.066537 | 2 |

5%只是描述性分桶，不是事后更改验收阈值。比值低于1表示新实现更快。

## 完整模型内替换

实际安装新v2到30层self/cross共60个attention入口；与canonical Torch dense B14相同的输入/历史/布局比较。每query两引擎各先做输出逐位warmup门检，再交错3轮各1个完整forward，含Python提交/布局/pack的整个模型查询。比较后全部有效KV/metadata未改变。没有Graph、静态prompt投影cache或新的共享prefix优化，不把它当最强B14系统加速基线。

| KV | query | Torch ms | 新v2 ms | 新/Torch |
|---|---:|---:|---:|---:|
| short | 0 | 310.782 | 229.751 | 0.7393 |
| short | 18 | 232.700 | 214.873 | 0.9234 |
| middle | 0 | 245.240 | 245.732 | 1.0020 |
| middle | 20 | 276.797 | 291.690 | 1.0538 |
| long | 0 | 448.631 | 478.481 | 1.0665 |
| long | 20 | 425.908 | 419.886 | 0.9859 |

六query中位时间求和比=0.969257，不是机器人E2E或有效提交成本ΣT_round/Σg。各case三样本很少、最小比值0.739不能作为稳定26%加速结论。已有静态prompt cache/shared-prefix强baseline不因本次对齐重新命名为新贡献。

## 覆盖、精度与资源边界

- 54点：三KV×B2 video/B2 action/B14 action×first/last×层0/8/29。B2/B14 action K288/1152/2160，video K272/1136/2144；B14为七独立Q16×CFG2，不跨候选注意。
- 684计时单元、18360 Event及等量wall样本，完整raw/逐轮CSV保留，无删慢点。前序同代码108attention/54block/六full30/三完整policy严格精度通过；本计时新进程输入同源CPU核对另列，不假称计时中重新验证了每个中间量。
- 用户确认空闲窗口，准入/收尾无他人可见context，但0MiB下NVML利用率25%仍未解释；不能把可见空闲升级成已证明物理宿主绝对无干扰。限一独立进程/三轮；性能证据仍须带资源和统计限制。
- 注意力kernel接近而block出现双向较大波动，需后续拆分host gap/clock/allocator/非attention算子；这些只是定位方向，当前没有根因证明，不再盲改数学内核。
- 完整30层是冻结verifier query；没有完整policy计时/机器人E2E/成功率新实验；项目四false和旧失败保持。

## 生命周期与收尾

本窗口r3+准入失败r4+r5=479.724秒；累计M22=1094.781/1200秒，剩余105.219秒。CUDA active UNKNOWN。新增源/编译/证据406.87MiB≤512MiB。r3/r5各return0并实际退出，r4整数秒下界错误在启动任何GPU child前失败，保留并计入0.140034379秒；不篡改为成功。未杀他人进程。

## 证据入口

[全部operator/block时间](PERFORMANCE-ONLY.csv) · [每轮](PERFORMANCE-ONLY-ROUNDS.csv) · [分组](GROUPED-RATIOS.json) · [完整30层](FULL30.csv) · [30层每轮](FULL30-ROUNDS.csv) · [汇总](RESULT.json) · [自有清理](../OWNED-CLEANUP-PERFORMANCE.json)
