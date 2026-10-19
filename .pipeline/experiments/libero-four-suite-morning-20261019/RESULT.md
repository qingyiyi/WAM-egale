# 10月19日两机晨间核验

Goal于08:41、Spatial于08:00完成各500组三臂，均为北京时间。完整机器人终态与组件审计分列。

|套件|臂|成功/500|成功率|800罚分均值|实际命令总数|严格组件通过/500|
|---|---|---:|---:|---:|---:|---:|
|goal|T|463|92.6%|158.806|79255|500|
|goal|C|464|92.8%|157.166|78439|335|
|goal|P|467|93.4%|156.466|78101|166|
|spatial|T|472|94.4%|145.998|72887|500|
|spatial|C|466|93.2%|152.964|76346|345|
|spatial|P|464|92.8%|155.304|77508|188|

T为Target，C为连续TF，P为同权重TF+PF2。

|套件|比较|救回/丢失|共同成功长度P50/P90/max|长度比>1.30|
|---|---|---|---|---|
|goal|T→C|13/12|1.0000/1.0442/2.6667|9/451|
|goal|T→P|14/10|1.0000/1.0518/4.8889|13/453|
|goal|C→P|13/10|1.0000/1.0440/4.2339|9/454|
|spatial|T→C|5/11|1.0000/1.0440/2.7905|18/461|
|spatial|T→P|7/15|1.0000/1.0330/4.1562|11/457|
|spatial|C→P|6/8|1.0000/1.0309/4.2000|15/458|

分位数使用线性Type7。全体800罚分/实际步数/自然失败全部保留，成功子集长度仅为补充。两套C/P均有FAILED_FIRST_TWO_S_PAIRING，不签严格配对/独立TEST/非劣；四项false保持。

并行质量运行中的C_update及ΣT_round/Σg原样保留于RESULT.json，只是探索性成本。正式E2E加速UNKNOWN，不能用target调用减少替代；所有臂使用既定512-token BF16优化reset，冷启动另列。

Object旧30局有完整终态，但新14条compact编号冲突、collector准入失败，仅前16条有效；用户已明确批准补14。新v3固定14补采全部完整，PACK_CURRENT准入通过（FIT6164/CAL3080）；固定ON3000训练已结束并启动全量评估，Matched Target200继续。09:21每卡一worker，其余分片因额外并发PID身份准入未通过仍排队。旧失败不改，原M17截止不重置；见OBJECT-RECOVERY-PLAN.md与RECOVERY-CAPTURE-LIVE.json。

本机昨夜CPU顺序B0通过；今晨GPU六真实fixture×两种缓存的66执行通过。A/A最大2.513%超过2.5%门限，计时停止，GPU0显存0且NVML24–25%异常；低开销监控等待状态改变。见../wam-block-gpu-followup-20261019/RESULT.md。

原始返回报告保留于raw/（约18MiB，不默认在GitHub重复发布）；RETURN-MANIFEST.json登记源路径/大小/时间。本文统计由summarize.py纯CPU独立核对1500唯一行/套、成败/步数/罚分与配对胜负，未重算原始velocity、严格初态或RNG。

# Object最新接续证据

## 09:19–09:21接续实证

固定14局已全部补齐，CURRENT_DATA原30局并集终态完成且guard=CLEAN。PACK_CURRENT原准入通过：FIT6164行、CAL3080行，all_episodes_terminal/all_rows_reached=true，counterfactual_columns_used=false、extra_teacher_forwards=0。ON3000已按固定500updates训练成功返回，控制器已进入RUNNING_OR_QUEUED_ALL_REMOTE_OBJECT500；FINAL_200_274已出现真实完整评估终态，不只是创建目录。

09:21巡检Matched Target为18/200，FINAL_200_274为5/225局（三臂合计，尚非5个共同完成组）；当前每卡一个模型worker，其他分片仍在队列。GPU工具报告PID与容器READY/OWNED的PID不相等，额外槽的observed.issubset(allowed)未通过；尚无可靠映射，不把未知PID标成自有、不放松原准入。PID-namespace只是待核机制假设，不能据此宣称发现他人进程。现有完整episode继续，不重启/截断。

按已有两套质量运行单局约60–73秒、当前两worker及阶段依赖，Object剩余墙钟仅能粗估半天至一天；尾部失败/更多并发会改变估算。原10/21绝对截止不变，到期不足则如实PARTIAL。所有并行计时仅探索性，四项false未改变。最新源指针：RECOVERY-PACK-TRAIN-SUMMARY.json、REMOTE-LATEST-AT-REPORT.json、REMOTE-WATCH-AT-REPORT.json、REMOTE-SLOT-ADMISSION-CHECK.json、PID-NAMESPACE-SOURCE-CHECK.txt。
