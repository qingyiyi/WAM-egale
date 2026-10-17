# 原 DEV96 的一次有界缺项接续

此目录是独立入口，创建源码不代表已准备或启动。原 `p0-tf-effect-dev96-v1-20261017-r1` 的源码、plan、FAILED 和已完成结果均不修改。只有原作业因原预算自然终止、独立时钟及严格自有进程清理完成后，才允许显式 prepare。

原完整机器人 success/horizon 终态全部冻结，包括算法失败。完整终态若缺 COUNTER、FINISH、命令、RNG 或 CAL 等审计证据，记 `TERMINAL_QUALITY_WITH_AUDIT_GAP`，保留结果并让最终结论为 PARTIAL；绝不重跑来替换其成败。只有机器人本身无完整终态的原 case 才进入 `pending_keys`。原 partial attempt 和资源成本保留。

兼容 plan 保持完整原96 `cases`，不改权重、seed、阈值、O3、horizon 或原 server/client。实际新增作业只按原 runner 的 arm→case 顺序执行缺失 M 局，跳过没有缺项的 arm；原生产 runner 的窄 AST 适配有逆向结构证明。新 COLLECTION/SUPERVISOR 成功标签为 `COMPLETE_PENDING_EPISODES_ONLY`，只能证明新 M 局，不伪称新跑96局。两 namespace 的完整96审计由独立汇总入口完成。

新所有 arm READY 必须与本轮原 T-READY 的真实初始化 RNG 相等。TF_ON/MLP_ON 使用新路径重新执行完整旧＋新 GPU CAL，不能复用旧 STARTED 目录冒充新执行。TF_CONT 保持原 source 的完整20237行既有证据复用，实际新执行行数0。

累计 target 上限仍是原280000，扣除原可知 COUNTER 总数，并对每个无 COUNTER 的已启动 attempt 保守扣4000。未知实际 forward 明确 UNKNOWN。新存储上限是原3GiB减旧作业实际占用，不复制旧终态或反事实数据。原耗时、等待、失败和中断全列入 `cumulative_budget`。

此次一次性授权的边界是北京时间2026-10-17：09:35停止工作、09:40清理GPU、09:55完成CPU汇总、10:00向用户汇报。不得更改原08:06/08:11守卫，不提前终止当前作业，不进行第二次接续或新训练。仅能使用固定新目录 `p0-tf-effect-dev96-continue-v2-20261017-r1`。

以下只是显式用法；本目录没有自动执行服务或定时器：

```sh
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 taskset -c 12 /usr/bin/python3 -B \
  投机采样/experiments/tf_effect_campaign_continue_v2/protocol.py --prepare \
  --job /rjs/cxz/论文探索/投机采样/results/runtime/p0-tf-effect-dev96-continue-v2-20261017-r1
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 taskset -c 12 /usr/bin/python3 -B \
  投机采样/experiments/tf_effect_campaign_continue_v2/launch.py \
  --job /rjs/cxz/论文探索/投机采样/results/runtime/p0-tf-effect-dev96-continue-v2-20261017-r1
```

接口：`predecessor_job`、`imported_terminal_manifest` 指向 `IMPORTED-TERMINALS.json`、`pending_keys`、`cumulative_budget`。manifest 的 `case` 是未加字段的原 case dict，`source_job` 单独记录；旧 case 的 campaign_id 保持旧 job，新 case 自属新 job。manifest 的 `COMPLETE_AUDIT` 只表示所需收据存在，不能代替独立完整状态/相机/数值审计。四项旧质量、TEST、非劣及正式性能权限始终 false。
