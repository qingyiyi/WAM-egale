# 首两S配对失败：既有完整episode审计的CPU定位

快照2026-10-19T18:12:38.267829+00:00，284完整episode审计中58条标记首两S失败；该数字随当前计时队列补齐更新，不是新的独立质量样本。0新GPU调用，未修改原FAILED。

OBSERVED：原审计从reset、S0_before、S0、S1按实际观察序列比较，只读取原布局全部有效KV，排除无效分配；初次调用前仅显式排除必被原infer重建的lazy timetable。失败并不是因为无效KV垃圾字节或计时字段被混入数学状态。原源身份见PAIRING-DIFFERENCES.json。

|最早失败记录|该记录首个显示差异路径|条数|
|---|---|---:|
|S0|`$/input/obs/observation.images.eye_in_hand_rgb`|27|
|S0|`$/phase_counters`|1|
|S1|`$/before/KV/0/k`|21|
|S1|`$/input/obs/observation.images.eye_in_hand_rgb`|9|

OBSERVED：本次最早失败记录类别为S0, S1，均早于第一次周期T；C/P的差异不能只归因于PF2刷新。差异路径类别按上表原样列出；它们支持“前两S尚未严格隔离”的原判定，不解释或删除任何任务失败。

UNKNOWN：compare_raw路径按字典遍历展示，首个显示路径不是因果根因；最早记录也不是全系统最早浮点差异。最多64个显示差异，truncated原字段保留。S0后的环境/compute-cache请求可能影响S1 before状态，但没有在此CPU审计中重放或重新核验这些中间输入，不能将其判为根因。

此处JSON包含原现场比较的结果/形状/路径，不包含被比较的原RGB或KV值；没有重新逐字节比较，也不能据此重算像素差幅度、teacher velocity或动作误差。是否其他原产物具备所需tensor仍须按具体路径核验。

RECOMMENDATION：保留6000局描述性成功率和当前工程成本，但不把PF2效果解释为已经隔离的因果收益、不签正式质量证书。后续若继续，先前瞻登记初态/渲染/缓存RPC输入的隔离证据，再考虑新的质量实验；不能追溯放宽旧门。
