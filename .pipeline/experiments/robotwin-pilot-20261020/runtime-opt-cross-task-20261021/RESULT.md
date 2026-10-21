# 当前工程组合跨任务/失败长尾核验

OBSERVED：四局完整终态，两组全程受用观察/物理/RNG/命令/server RPC摘要、计数与终态相同；原源码未改。此次为原Target工程重复，非新独立质量样本。两臂固定顺序、每case各一次，计时为探索性客户端生命周期。

|场景|原版/组合成功|原版→组合秒|耗时下降|双方提交/真实推进|双方罚分|
|---|---|---:|---:|---:|---:|
|place_object_basket/valid14|True / True|177.761 → 166.753|6.19%|240 / 235|235|
|open_microwave/valid14|False / False|921.867 → 887.386|3.74%|1520 / 1500|1500|

basket两臂成功；microwave两臂均自然算法失败，失败保存，不能以耗时变短称质量改善。此前microwave valid16组合−7.33%与本轮分列，不将三个case平均外推完整套件收益。

GPU生命周期2197.362442秒；原M40累计54718.243980/86400秒。guard/children自然退出，remaining_owned为空，远端两卡核查无计算进程。原01:26截止保留；无自动后继，1800局/新增160GPU小时未启动/未批准。

UNKNOWN：完整六任务、不同失败路径、重复计时波动及双卡扩大吞吐。摘要保存不等于持久化全部原图/KV。四科学权限false。

RECOMMENDATION：当前组合在三个曝光场景有有限等价证据，可继续作为显式工程候选；完整扩大时各算法臂使用同等工程配置并继续审计。全局默认未修改。

证据：STATUS.json、COSTS.csv、两份*-AUDIT.json、各完整局STATUS/OBS-AUDIT/COMMAND-STATE-AUDIT/commands及server审计。所有旧FAILED/PARTIAL和累计保留。
