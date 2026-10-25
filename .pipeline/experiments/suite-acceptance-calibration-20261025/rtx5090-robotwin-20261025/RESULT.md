# 5090 RoboTwin阈值DEV有限结果

状态PARTIAL_NO_WHOLE_EPISODE_BUDGET：固定六case中仅完成place_object_basket/valid49的一组四臂，未完成全panel。不续期、不自动重跑。截止与原失败保留。

|臂|自然终态|环境take_action计数|提交命令|失败罚分|
|---|---|---:|---:|---:|
|T|失败|700|720|700|
|OLD|失败|700|720|700|
|Q75|成功|265|272|272|
|Q90|成功|384|400|400|

四臂同5090严格物理/观测/RNG初态核查均通过；历史A100像素不一致单列。全程full512 BF16 reset bank，单worker实际峰值reserved约22.13GiB，32GiB卡不并两个worker。4193 teacher尝试、68 policy。

这只是一个已暴露DEV，不能选择最终tau或称总体成功率改善。新T失败与旧M41成功不一致仍待核，不能借旧T替换。四科学权限仍false。失败罚分为原Robo任务上限700，不套LIBERO800，也不把720已提交命令改成700。

后台watcher已验证owned退出、归档212文件/60309007B；5090临时根（含权重/环境/资产）已删除，算子workspace保留。证据：WATCH-STATUS.json、ARCHIVE-MANIFEST.json、LIVE-SNAPSHOT.json及raw/，不把旧SERVING日志当作进程仍存活。
