# RoboTwin固定采集和后继训练晨间结果

更新：2026-10-21T09:13:44.589629+08:00。

## OBSERVED

- Target48原采集：41成功、7自然失败；不改变原分区和证据。
- 用户新授权的place_object_basket/valid08补采于08:42:40开始，08:50:21清理，生命周期461.002秒；23 policy chunks、1159次teacher，720实际命令，任务horizon/失败罚分700。完整自然失败，不能把它删除或写成基础设施PARTIAL；原0动作PARTIAL同时保留。
- 固定PARENT18现18/18齐：13成功、5自然失败，282 policy chunks、14011次teacher。与Target48不是相同初态面板，不据此比较成功率。
- 完整监督审计通过：FIT12/6269行，CAL6/3078行；实际到达root/selected parent query，完整teacher重查询条件仍不齐。
- CHILD2500与CONT3000已各完成固定500 updates；同normalizer/损失/种子和原配方，最终checkpoint，无CAL选权重。
- 首个训练启动曾在解析入口时因缺右括号退出，0 updates、0 teacher；原记录保留。修复仅语法，独立r2首次完成数学训练，成本计原账。
- 两个权重分别在已有Target CAL12和PARENT CAL6共10771 query完成GPU predict/decode BF16逐位一致核验；RNG不变、tau固定为0.007879490032792091，没有重调阈值。
- 原累计GPU生命周期现5.7624/24小时；新单局60分钟上限没有用于延长其他工作。

## UNKNOWN / RECOMMENDATION

GPU adapter/decode一致不等于B2/B14 teacher一致或闭环质量通过。CURRENT18、ON3000、固定三臂DEV90仍未启动。
接下来是固定CONT3000的CURRENT18采集→审计→ON3000→已登记DEV；不能用当前CAL代理替代完整episode。
当前后继链09:30截止，开局须预留1800秒整局和900秒清理，已不足完整预留，因此没有启动会越过原截止的新episode。
需下一执行窗口接续；不改旧deadline、不自动重试或缩短episode。108完整512-token BF16 reset bank仍启用；RoboTwin完整episode bank等价未另测。

四项科学权限仍false，尚无RoboTwin新TF最终成功率/加速结论。

## 原件

- `overnight-remote-evidence/parent-valid08-authorized-retry-20261021/STATUS.json`
- `overnight-remote-evidence/parent-audit-complete18-20261021/RESULT.json`
- `overnight-remote-evidence/branches-after-fixed18-20261021/STATUS.json`（语法失败）与`branches-after-fixed18-20261021-r2/`（训练）
- `overnight-remote-evidence/branch-cal-20261021/`（固定接口CAL）
- 远端权重：`/root/cxz/robotwin/runs/branches-after-fixed18-20261021-r2/{CHILD2500,CONT3000}/checkpoint.pt`。不公开上传权重。
