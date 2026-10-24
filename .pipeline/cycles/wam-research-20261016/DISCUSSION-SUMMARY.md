# 当前讨论与采用边界

用户主线是投机改变小action工作负载，为候选感知tile/warp调度创造机会；不是把普通batch/cache命名为新MegaTransformer。当前单层A保留，FFN有限候选负结果保留，旧global-stage同步不自动恢复成完整DAG。本轮M41只测既有投机路径，未接入单层算子或证明H1–H3因果增量。

M41观察到action单位成本下降，但任务结果有suite差异。Long/Spatial/Object连续TF约1.06–1.12倍client任务收益；Goal连续TF因失败仅0.83倍而PF2约1.12倍；Robo两臂约0.93倍且额外失败。PF2在Spatial/Object有1.70/1.89倍成功长尾，不能普遍开启。Robo首个S相同但第二S已在PF2之前分歧，先诊断，不能据此追加训练猜根因。

TF结构仍可暂冻为研究接口，不代表当前权重/接受器已全局验收。全部四质量/正式性能字段false；不改旧FAILED/retention、不混DEV与TEST、不把call减少当E2E、不把7候选当Q112、不猜未保存velocity/KV。后续先完整A100质量队列及微波炉既定诊断，正式计时须另有质量/资源放行。

本机现为轻量开发/证据，双A100接续质量，5090用于性能/算子。M41新资产测完核验回传即清空5090，下次重传；原operator环境保留。A100原资产/可复用导出包与冷归档保留；所有BAAI写入限~/cxz。旧预算起点/硬截止不重置。
