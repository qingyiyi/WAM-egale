# 四套新独占工程计时

状态：PARTIAL_ENGINEERING_TIMING，237/360完整局；来源 2026-10-19T17:29:54.817798+00:00。

固定10任务/init49，三次独立server进程重复。计时重复不增加独立质量N；原6000局成功率另见RESULT.md。四质量权限保持false，原FAILED/训练曝光/组件审计与长尾不改。

| 套件 | T/C/P完整数 | T/C/P成功数 | TF暖episode | PF2暖episode | TF有效更新 | PF2有效更新 |
|---|---|---|---|---|---|---|
|long|5/4/5|5/4/4|1.3617×|0.8437×|1.6589×|1.3355×|
|goal|30/30/30|29/30/29|1.5232×|1.2314×|1.5952×|1.3385×|
|spatial|30/30/30|26/27/27|1.4214×|1.3206×|1.5446×|1.3312×|
|object|14/14/15|14/14/15|1.2330×|1.1136×|1.5741×|1.3410×|

上述比率均为已完成同case、同重复的时间总和之比。未齐套件为PARTIAL，不能当完成主表；自然失败时间也在全样本分子/分母。双方成功子集、实际步数/800罚分及每次重复分列JSON和CSV。

有效更新成本=ΣT_round/Σg。GPU0 Goal/Object、GPU1 Spatial/Long；不跨卡拼接同case。原COUNTER中AutoDL/exploratory字符串原样保留，实际硬件/隔离来源以新登记和NVML观察为准。

warm episode从环境构建后、reset前计到自然终态，含初始化/稳定、reset/policy/cache/环境，排除env.close与证据落盘。client_total还包含入口导入、环境构建、env.close及raw证据写入，不能当纯任务E2E。模型冷启动是每phase共享setup，另列，未测三臂分别cold端到端。详见TIME-BOUNDARY.md。

Object repeat1入口在任何server/episode前被旧Goal/Spatial白名单拒绝，0teacher/0局，保留且不重试。仅接续原先未启动的Object repeats2/3；原360预期保持，最大330完整，整面板按PARTIAL报告，不把缺30局计为算法失败。
