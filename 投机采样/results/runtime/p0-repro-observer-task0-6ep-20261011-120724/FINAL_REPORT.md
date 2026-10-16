# 六局种子/observer完整策略复现关卡结果

完成：2026-10-11T12:41:37.364130+08:00。同一S0+冻结target，OFF-A/OFF-B/ON，各两个独立EGL worker首局init3/4，完整horizon；无teacher/训练/RL/联训。公共源码不改，job-only种子/状态指纹；不是clean timing。

| 条件 | init | success | env steps | chunks |
|---|---|---|---|---|
| repeat_off_A | 3 | True | 293 | 19 |
| repeat_off_A | 4 | True | 600 | 38 |
| repeat_off_B | 3 | True | 293 | 19 |
| repeat_off_B | 4 | True | 600 | 38 |
| observer_on | 3 | True | 293 | 19 |
| observer_on | 4 | True | 600 | 38 |

OFF/OFF在测试inits的测量稳定性：True。详见reproduction-comparison.json，所有第一处分叉保留。若OFF/OFF有差异不能把ON差异归因observer；相同已测字段不等于完整隐藏状态/噪声恢复，也不证明旧失败根因。保存action/video输出张量需要后续CPU逐位复核，自动报告不冒充该复核。

下一步：读数据并CPU对齐最早分叉；通过后再单独同负载T/A_off成本计数。不自动加局、训练、RL/KL或大小模型联训。
