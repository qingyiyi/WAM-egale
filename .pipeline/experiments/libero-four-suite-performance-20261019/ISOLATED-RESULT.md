# 四套新独占工程计时

状态：PARTIAL_ENGINEERING_TIMING，301/360完整局；来源 2026-10-19T18:36:02.930342+00:00。

固定10任务/init49，三次独立server进程重复。计时重复不增加独立质量N；原6000局成功率另见RESULT.md。四质量权限保持false，原FAILED/训练曝光/组件审计与长尾不改。

| 套件 | T/C/P完整数 | T/C/P成功数 | TF暖episode | PF2暖episode | TF有效更新 | PF2有效更新 |
|---|---|---|---|---|---|---|
|long|20/21/20|20/19/18|1.1023×|1.0080×|1.6286×|1.3518×|
|goal|30/30/30|29/30/29|1.5232×|1.2314×|1.5952×|1.3385×|
|spatial|30/30/30|26/27/27|1.4214×|1.3206×|1.5446×|1.3312×|
|object|20/20/20|20/20/20|1.2162×|1.0993×|1.5748×|1.3411×|

上述比率均为已完成同case、同重复的时间总和之比。未齐套件为PARTIAL，不能当完成主表；自然失败时间也在全样本分子/分母。双方成功子集、实际步数/800罚分及每次重复分列JSON和CSV。

有效更新成本=ΣT_round/Σg。GPU0 Goal/Object、GPU1 Spatial/Long；不跨卡拼接同case。原COUNTER中AutoDL/exploratory字符串原样保留，实际硬件/隔离来源以新登记和NVML观察为准。

warm episode从环境构建后、reset前计到自然终态，含初始化/稳定、reset/policy/cache/环境，排除env.close与证据落盘。client_total还包含入口导入、环境构建、env.close及raw证据写入，不能当纯任务E2E。模型冷启动是每phase共享setup，另列，未测三臂分别cold端到端。详见TIME-BOUNDARY.md。

Object repeat1入口在任何server/episode前被旧Goal/Spatial白名单拒绝，0teacher/0局，保留且不重试。仅接续原先未启动的Object repeats2/3；原360预期保持，最大330完整，整面板按PARTIAL报告，不把缺30局计为算法失败。

## 双方成功与固定chunk口径

|套件|TF双方成功N / 暖episode|PF2双方成功N / 暖episode|TF完整policy/chunk|PF2完整policy/chunk|
|---|---|---|---:|---:|
|long|18 / 1.3368×|18 / 1.2239×|1.3645×|1.2196×|
|goal|29 / 1.3194×|28 / 1.1620×|1.3581×|1.2158×|
|spatial|23 / 1.2531×|23 / 1.1699×|1.3279×|1.2126×|
|object|20 / 1.2162×|20 / 1.0993×|1.3412×|1.2160×|

固定chunk比率是各臂Σ完整policy时间/Σpolicy chunks之比；自然观察与历史不同，不能称同一输入chunk重放。双方成功子集只作补充，上表全体完整尝试仍保留失败时间。

## 组件门与测量覆盖

|套件|组件审计终态计数|round/action计时覆盖齐全|
|---|---|---|
|long|PASS_COMPLETE_EPISODE_COMPONENTS: 57, FAILED_FIRST_TWO_S_PAIRING: 4|True|
|goal|PASS_COMPLETE_EPISODE_COMPONENTS: 60, FAILED_FIRST_TWO_S_PAIRING: 30|True|
|spatial|PASS_COMPLETE_EPISODE_COMPONENTS: 65, FAILED_FIRST_TWO_S_PAIRING: 25|True|
|object|PASS_COMPLETE_EPISODE_COMPONENTS: 60|True|

FAILED_FIRST_TWO_S_PAIRING/FAILED_COMPONENT_AUDIT原样保留，包含在完整尝试表中。资源巡检未观察到同卡额外模型进程不等于算法、首两S或质量门通过；本表仅为未准入候选的工程成本观察，不能签正式性能、独立TEST或闭环质量保证。

## 独立server进程重复的波动

|套件/方法|完整计时重复数|暖episode速度比均值±样本SD|完整policy/chunk速度比均值±样本SD|
|---|---:|---|---|
|long/C|2|1.1028±0.0538|1.3643±0.0064|
|long/P|2|1.0079±0.0220|1.2195±0.0054|
|goal/C|3|1.5236±0.3281|1.3561±0.0179|
|goal/P|3|1.2806±0.4373|1.2181±0.0301|
|spatial/C|3|1.6158±0.8875|1.3380±0.0484|
|spatial/P|3|1.3211±0.2310|1.2122±0.0065|
|object/C|2|1.2162±0.0066|1.3412±0.0073|
|object/P|2|1.0993±0.0040|1.2160±0.0050|

此处是同一10case子面板的进程级计时重复，Object仅两轮；不是三轮各500局成功率，也不是新增独立初态。均值是各重复速度比的算术平均，上方主表是匹配时间总和之比，两者口径分开。三或两重复的样本SD仅描述当前波动，不冒充总体置信区间。
