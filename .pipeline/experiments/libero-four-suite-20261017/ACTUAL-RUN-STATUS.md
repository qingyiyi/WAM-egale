# M17实际执行索引

核对：2026-10-17 18:42 北京。新三套模型/机器人episode/训练仍0；控制器等待不是已开始训练。旧M16/4f04、FAILED、NO_GO、四false不改。

|工作|实际终态或状态|证据|
|---|---|---|
|Object/Goal资产|远端完整，各10,177,831,668B；本机Object完整副本已装配；非官方community Target|asset r2；local asset admission r1；固定revision见PLAN|
|完整reset bank|Goal/Spatial CPU2455.0秒成功；Object远端v3 r3 CPU1220.5秒成功；全部512 BF16，尚未在新suite在线reset重新审计|[真实远端快照](../../../投机采样/results/runtime/p0-tf-suite-cache-rotation-inputs-cpu1-v2-20261017-r1/ACTUAL-CAPACITY-AND-PLACEMENT-r1.json)|
|三条新远端pipeline|8eaf130，共享GPU/CPU原子租约；Object32543、Goal31852、Spatial31853，等待原Long实际清理/Spatial资产|[新登记与现场](../../../投机采样/results/runtime/p0-tf-suite-remote-placement-v2-20261017-r1/ACTUAL-20261017-1830.json)|
|空间轮换调度|37f15d1、PID32254；等待Long500全终态＋owned/guard实际清理＋文件无消费者，尚未删权重|同一快照；`p0-tf-suite-cache-rotation-cpu1-v2-20261017-r1/status.json`|
|Long M16|双卡四worker继续；新285闭合局＋原69＝354/1500，另有组内运行局；不是完整质量结果|同一现场快照；[原索引](../libero-fullscale-20261017/ACTUAL-RUN-STATUS.md)|
|本机清理|7个已停失败下载残片，985,044,916B（约0.917GiB）；完整权重、失败记录与原始结果保留|[逐文件清理结果](../../../投机采样/results/runtime/p0-tf-suite-abandoned-download-cleanup-cpu1-v1-20261017-r1/RESULT.json)|

Object最终准备job现为`/root/wam-suite-research-20261017-v3/object-chain-r1`，占现有OS盘、5GiB上限和实际空闲≥5GiB；原数据盘R2准备0局终态保留。Goal/Spatial仍数据盘原job。Long完整归档producer128d435已在本机CPU1启动等待（PID443492）：全部原件完整回传本机后，才退远端重复payload和明确登记的COUNTER副本，其他JSON/FAILED不动；需先等Long及轮换审计完成，当前归档状态WAITING、复制0/删除0。所有原件在本机保留，登记定位映射，不能宣称凭size校验证明独立密码学身份。

## 本轮资源修订与原记录

用户要求本机减负，继而明确暂不扩容。新三套零新增数据前，Object从本机前瞻迁至AutoDL GPU0，Goal同GPU0，Spatial GPU1。每套500case的所有臂固定同物理卡；最终两shard共享同卡最多2模型槽位，全局flock＋PID出生身份＋实际清理释放，CPU0–15放server/client，16–19共享训练/控制，物理CPU池不超20。

原9aa准备控制器Object本机r1/r2、Goal/Spatial远端r1分别记录替换原因并退出，没有新增模型或episode，不覆盖其FAILED/PARTIAL。新namespace保持原17:11:31起点、10/21 16:56停新/17:11清理。源8eaf只改变资源调度和独立report_v2选择，原模型/数据/数学/阈值/树/bonus/训练固定预算不变。

原本机bank CPU7被3590秒外层timeout终止、exit124，无完整新bank；加EXTERNAL-TERMINAL，原状态不追改。Object远端v3 r1缺历史bank收据，r2启动参数错误均在编码前失败；补真实原收据/纠正启动后独立r3沿r1原2h截止完成。原stream import r3总3200秒TimeoutError，只有Object完整，Goal残片后来经独立获准清理；不能称完整导入两套。旧bridge已因资源安排替换退出，未执行其Object缓存删除/Spatial下载。

当前本机余约94.9GiB；远端数据盘约15.9GiB、OS盘10.98GiB。没有立即可删的大块废弃远端数据。Long完整结束后，只在完整本机原件/固定来源与3分片stat、全部进程/FD/maps核实后回收可重建远端Long Transformer缓存，预计9.48GiB；保留config/index/T5/VAE/tokenizer与Long全部原始结果。实际释放量由UNLINK/statfs收据记录，未知时不删。随后fresh Spatial下载保留10GiB空闲线并派生原scaffold绑定。若不能安全释放或空间仍不够，BLOCKED，不自动扩盘或降低保留线。

## 固定方法、预算和下一步

[完整PLAN](PLAN.md)。各套独立TF64/seed20260923、Target200=FIT180/CAL20；PARENT2000采30，再CHILD2500/CONT3000，CONT采新30，ON从同CHILD+500。无结构/seed/阈值搜索。最终500case×T/C/P，Target200只读复用，另训练采集60局，1560新增/套、4680总局。FIT/CAL及init26–49训练未见子集明示，非独立TEST。

原96h与18,723,000尝试/15GiB研究证据帽不重置；原Long仍10/19 13:40停新/13:55清理。每局horizon800、1200秒、4000尝试，等待/准备/失败/清理计账，无自动重试。全成败、实际命令（如796）与800罚分分列。并行C_update/ΣT_round÷Σg/policy/episode只作探索性成本，formal E2E仍UNKNOWN；旧fixture缺口不借compact表冒充修复。

NEXT：监督Long及bank实际终态；Long清理后Goal/Object直接完整T_POOL，轮换Spatial后接入同流程。当前完整新增工作粗估48–72小时，首批新suite任务后按实测修订，96h为止损帽。10/22 go/no-go、10/23冻结、10/28主表、11/01 18:00中文证据包；不自动120/TEST/正式性能。
