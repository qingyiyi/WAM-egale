# 两站80局独立描述与恢复重复性小结

原远端FAILED/无RESULT及原aggregate PARTIAL保留；四项权限均false。本表不是原终态或质量门修复。

| 方案 | 成功/20 | 全部真实步数 P50/P90 | 失败800罚分均值 | target/chunk |
|---|---:|---:|---:|---:|
| T | 19/20 | 253.0/294.1 | 278.30 | 73.895 |
| TF_CHILD | 19/20 | 252.0/387.0 | 285.45 | 43.408 |
| TF_CONT | 20/20 | 255.5/288.0 | 259.15 | 43.524 |
| MLP_CHILD | 20/20 | 248.5/316.5 | 258.50 | 43.674 |

主候选 TF_CONT/T：20固定配对中19局双方成功，步数比 P50/P90/max=1.004184/1.032907/1.172996。数值描述筛选六项均true，20对不能证明3pp非劣。

全部60组物理/target-RNG逐位一致；相机strict56/60。TF_CONT/T主比较物理20/20，相机strict19/20。4差异全部为task2腕相机：本机child2pixel/6channel、CONT1pixel/3channel、MLP1pixel/3channel；远端child1pixel/3channel，max均1。不得删除这些组。

恢复重复性固定13局：初始物理13/13、target RNG及reset13/13、相机11/13、完整commands10/13、success12/13、步数10/13一致。旧13不混入80。
具体不一致见独立13局 SUMMARY.json；差异仅描述，不能作为具体失败根因。原Python/NumPy RNG对象未保存，只能比较记录hash。

本机40完整saved值实际本地复核；远端40完整2070PT/22334数组由实际远端CPU回执补核，本地仍未读664份远端obs值。见SUPPLEMENT.json映射与来源。
本轮80重建CPU 14.793s，恢复13核对CPU 2.635s；无GPU/SSH/forward/新数据/TEST。
