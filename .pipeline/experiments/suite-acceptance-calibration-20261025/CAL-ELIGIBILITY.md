# 五套已有 CAL 资格与冻结候选

OBSERVED：五套完整 CAL 部署数值重放完成；主池共21045条，次池共61985条，总83030条。所有完整episode级FIT/CAL交集为空；三个小LIBERO套件采用fixed final update，未用CAL选择checkpoint。每池前16个query的production adapter与原始decode逐位相同，RNG无变化。GPU作业166.03秒生命周期，0 teacher/0机器人episode。

| 套件 | observer/target CAL行数 | 旧tau | Q75 | Q90 | 旧tau主池query覆盖 |
|---|---:|---:|---:|---:|---:|
| long | 6805 / 20237 | 0.00783968053 | 0.0025799477 | 0.00758647779 | 90.51% |
| goal | 3808 / 13573 | 0.00783968053 | 0.0031982786 | 0.00904507004 | 87.53% |
| spatial | 3850 / 11417 | 0.00783968053 | 0.00316868932 | 0.0103148837 | 86.33% |
| object | 3080 / 9065 | 0.00783968053 | 0.00315794139 | 0.00911117718 | 87.69% |
| robotwin | 3502 / 7693 | 0.00787949003 | 0.00490120891 | 0.00907344092 | 87.51% |

主校准固定采用CONT3000产生的OBSERVER_CAL，冻结ON3000权重在原state/previous/sigma上重放。它是FIT隔离的工程CAL，但不是当前ON3000的on-policy轨迹，不能从覆盖率推断在线树接受率或SR。Long/Robotwin旧输入没有逐行dtype标记，按原BF16收集契约重放；存储值可精确表示为BF16，但不能补造原标记。旧Long target权重逐字节身份缺失，TARGET_CAL仅作次分布审计。

阈值按task等权、episode等权、query等权的两头最小score加权逆CDF取75%/90%。FP32打分，BF16方向边界保持；不是每个task调参。DEV开始前冻结所有值，不因成功率结果修改。旧全局默认/原失败记录/全部四项false保持。

完整DEV采用同卡新Target、OLD、Q75、Q90，PFoff、固定权重/normalizer/引擎、full512 BF16 reset bank。有限panel是已暴露DEV，不是独立TEST；只有完成且通过原质量要求才可建议进一步投入，不能以CAL或调用数宣称闭环改善。

UNKNOWN：新阈值的任务成功、长尾、实际计算成本，等待完整DEV。原双A100队列不修改。

来源：SOURCE-AND-DATA-AUDIT.json、calibration/*-RESULT.json、CAL-PROCESS-RESULT.json、CALIBRATION-RULE.md。
