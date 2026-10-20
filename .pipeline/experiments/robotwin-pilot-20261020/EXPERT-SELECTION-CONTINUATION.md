# 专家筛选中断与按原规则接续

2026-10-20。prefilter-control-6用时790.555秒后终态FAILED_OR_PARTIAL，清理完成。原件保留；Target等待器first6-1仅因依赖失败停止，0模型/episode/GPU尝试。不能把此时60个有效初态说成114个清单完成。

前三任务各19有效初态；lift_pot已3个有效、8个不可解，seed10011在官方`grasp_actor`触发`target_pose cannot be None for move action`。源码链：

- `envs/_base_task.py:1096–1103,1134–1137,1165`：找不到可用抓取姿态时返回None/None。
- `envs/_base_task.py:1196–1206`：None与None相等，随后构造move Action。
- `envs/utils/action.py:76`：move不接受None。
- 固定LingBot `evaluation/robotwin/eval_polict_client_openpi.py:479–492`：官方专家筛选捕获专家异常，seed加一继续；未运行模型策略。

因此这是有明确源码依据的专家不可解初态，不是新OIDN/驱动故障或模型失败。先前保守wrapper未覆盖这个官方筛选分支，导致停止。接续仅白名单此精确断言及grasp_actor栈；其他基础设施错误仍停，不照抄泛化catch-all。

`prefilter_continue.py`从已有60条开始，引用原INITIAL.pt，不重跑10011或前三任务，不重新抽seed/指令/划分；将10011作为原官方规则跳过，从lift_pot10012继续。新输出prefilter-4/control-7保留来源；旧FAILED不改成PASS。

原19有效seed/任务、6任务、1200总专家尝试和3600秒筛选绝对截止不重置，worker3300秒自原开始累计。原GPU累计账本追加真实新进程占用。Target-first6-2只接续新清单，沿first6-1的原绝对deadline，不重置7200秒。没有重复完整机器人样本或训练，不需要新增模型测试预算。
