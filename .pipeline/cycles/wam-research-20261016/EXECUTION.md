# 最近执行检查点

## 2026-10-17 21:31 北京：旧AutoDL完成阶段收尾，正在完整备份

用户明确要求阶段结果后暂停旧机，准备换新服务器。Maintain已保存对应Cycle约束；研究范围、原预算和四false不改。[当前换机计划](../../experiments/autodl-stage-migration-20261017/PLAN.md)。

fb519d2独立控制64460/birth6697066812，先暂停Goal/Spatial排队（PAUSED_PARTIAL、0局、租约已释放）、停止轮换（0删除）、关闭旧全1500归档等待器443492（原BLOCKED/PARTIAL保留，远端RPC实际exit0）。只冻结旧CPU dispatcher，客户端/模型自然运行；固定已进入的四case181/182/192/193，以原d5448bb客户端补完同case未运行臂后收尾。旧科学源码及CASE/FAILED不改。

阶段已实际暂停：189完整三臂组/567局，311未开始组，partial case0、无截断；原Long supervisor仍PARTIAL_OR_FAILED_LONG500_COVERAGE，guard和实际owned组已清理。不是完成500组，更不是新质量证书。实例尚未关机。

d098fd2 CPU1流式备份已启动：本机604568/birth1351289843，`p0-autodl-migration-backup-local-cpu1-v1-20261017-r1`。预计37349个文件/1,970,600,995B，含所有remote runtime原件、选定TF/normalizer、全部bank、失败及小配置；无远端临时tar、无SHA、无删除。原Target大权重不复制，保留本机原件和固定重建来源。9000秒总生命周期、copy≤5400秒、payload≤4GiB、单文件64MiB/manifest32MiB、磁盘余≥5GiB。当前COPYING，完成且CLEANUP后才确认旧机可释放；新机SSH信息未提供，未启动迁移或新作业。

[暂停前完整执行记录](history/EXECUTION-before-stage-migration-20261017.md)逐字保留；旧文相对证据路径按原Cycle目录解释。原预算：Long10/19 13:40停新/13:55清理；M17 10/21 16:56/17:11，不因暂停换机重置。

## 2026-10-17 21:59 北京：最新授权接续；备份改大块流

用户最新允许新机信息提供前推进已有授权工作，Maintain已保存约束决定。原567局暂停终态不动，备份后拟独立namespace接续311未开始组，按10组三臂/30局屏障暂停；四false/数学/优化reset/原48h总限不变。2b1f039独立wrapper复用d544原server/client/Runner.run_case，旧166＋更早23组以多代源映射保留，不用二代report遗漏23组。仅本机静态源码审查，无smoke/独立测试/SHA；尚未启动新GPU。

备份v1 604568实际SIGINT退出130，原FAILED/CLEANUP保留，完整复制12751文件/574337066B（部分文件副本另计），SSH闭合、源删除0。v2源9f8f251，635195/birth1351410862，从单tar大块回传并现场核对全部entry/字节数；原deadline1792252790.1893034不重置，旧部分副本累计存储帽内保留。当前仍COPYING，不能据此宣称备份完成或旧机可释放。

## 2026-10-17 22:04 北京：原件备份与阶段统计完成，新接续已启动

9f8f251备份v2实际exit0/CLEAN，650.704秒完成37349文件、源1970600995B、tar2053969920B；所有条目/字节数与源前后stat核对，未做SHA，旧源删除0。原v1失败和594560318B部分副本保留，仍在累计4GiB预算内。selected TF/normalizer/所有bank和失败原件均归档。原deadline未重置。

7ca036a本机CPU1阶段统计已完成/CLEAN：T183/189、C185/189、P181/189，所有自然失败纳入。C/P相对T最大共同成功长度比1.7004/1.5465，组件FAILED分别15/27保留；不是质量通过或独立TEST，不升级四false。共享探索ΣT_round/Σg：T0.135247、C0.054411、P0.080197秒，不冒充正式E2E。统计原件p0-long-stage189-descriptive-local-cpu1-v1-20261017-r1。

用户最新允许新机未接入前继续。2b1f039已在独立p0-tf-long-shard-resume-v1-20261017-r1启动，supervisor69459/birth6697299895、runner69462/birth6697300590、guard69461/birth6697300570。当前4模型装载中、0新局，非READY或已完成。原d544数学server/client/Runner.run_case原样复用并核对源码bytes，无smoke/独立测试/SHA。只明确311未开始case，保留原189源映射；10组三臂/30局屏障处读STOP_AFTER_SHARD。所有旧attempt累计debit591787；原6001000/10GiB及10/19 13:40/13:55不变。首分片185/186/189/190/191/194/195/196/197/198。

新结果未在旧567归档内，迁移前需增量备份和真实清理；不可直接关旧机。Goal/Spatial仍暂停，本机Object原超时不自动重试。新机SSH尚未收到。登记与实际部署收据：p0-autodl-migration-stage-plan-v1-20261017-r1、p0-tf-long-shard-resume-deploy-local-cpu1-v1-20261017-r1。

22:08实际只读核验：四worker全部READY，首SHARD-000已在完整episode运行（case186/C、196/P、197/T、185/P）；两卡GPU利用率98%/97%、显存约29GiB/卡，无新FAILED。实际证据p0-tf-long-shard-resume-observe-local-cpu1-v1-20261017-r2/READINESS-SUMMARY.json。尚未把运行中的分片写成完成。
