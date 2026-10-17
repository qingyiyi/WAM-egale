# M17实际执行索引

核对：2026-10-17 22:05 北京。旧AutoDL按换机请求实际完成阶段暂停；用户随后允许新机接入前推进可做工作。旧567局全原件已备份完成，新namespace四模型全部READY，首完整任务分片在跑，尚未签完整分片终态。新机SSH未提供；新结果须增量备份后才能关旧机。

|工作|实际状态与边界|证据|
|---|---|---|
|Long M16|旧189完整组/567局、311未开始、partial0；原supervisor PARTIAL保留。新2b1f039独立p0-tf-long-shard-resume-v1-20261017-r1已启动，10组三臂/30局屏障；4模型READY、完整任务运行中，不重复旧组|[阶段汇总](../autodl-stage-migration-20261017/STAGE189-RESULT.md)、原v2 job、resume实际收据|
|Long阶段描述|Target183/189、TF185/189、PF2 181/189；原组件FAILED/初态缺口保留，最大长度比超旧门；包含训练暴露，四false不变|[逐局汇总](../autodl-stage-migration-20261017/STAGE189-RESULT.md)|
|Object/Goal资产|远端各10177831668B完整；本机Object完整；非官方community Target|固定来源见PLAN；原asset r2与local admission|
|完整reset bank|Goal/Spatial CPU2455秒、Object远端v3 r3 CPU1220.5秒完成；本机复用Object bank128.9秒、原512 BF16，无T5/teacher重做；新suite在线完整任务等价仍未签证|原bank与导入RESULT|
|Object本机v4|READY/setup749.6秒；首task0/init0触发1200秒cap，0完整局；20:24 FAILED_OR_PARTIAL_NO_RETRY、guard CLEAN/owned组空；不记自然算法失败，不自动重试|[v4资源登记](../../../投机采样/results/runtime/p0-tf-suite-local-placement-v4-20261017-r1/AMENDMENT.json)|
|Goal/Spatial原远端pipeline|8eaf130、31852/31853均PAUSED_PARTIAL/0完整局、训练0，租约实际释放；旧job不重启|p0-autodl-migration-stage-pause-v1-20261017-r1/PENDING-WORK-HELD.json|
|缓存轮换|37f15d1已停止，BLOCKED、deleted_cache_files=[]，无asset child；未删权重、未下载Spatial|原rotation STOP/FINAL-OWNED-CLEANUP|
|旧全1500归档等待器|443492实际结束，原BLOCKED/PARTIAL保留；远端RPC32756实际exit0，复制0/删除0|旧archive RESULT/CLEANUP|
|阶段全原件备份|9f8f251、635195实际650.7秒完成/CLEAN；37349文件/1970600995B，tar2053969920B；含TF/normalizer/bank/全部失败，旧源0删除、无SHA。v1回传慢SIGINT130及部分副本保留|p0-autodl-migration-backup-local-cpu1-v2-20261017-r2/RESULT.json|
|项目空间清理|本轮只清7个已停止失败下载残片985044916B；完整权重、失败记录和唯一结果保留|原abandoned-download-cleanup RESULT|

## 执行与下一步

最新接续source2b1f039、supervisor69459/runner69462/guard69461，旧attempt累计debit591787；原M16起点10/17 13:55、10/19 13:40停新/13:55清理、6001000attempt/10GiB不变。共享GPU质量耗时仅探索性；同case三臂仍同卡，原d544数学与优化reset不改。收到新机SSH后在完整30局屏障收尾、增量保存新原件、清理后迁移，不因换机重置账本。

三小套件研究授权仍有效，原M17起点10/17 17:11:31、10/21 16:56停新/17:11清理，4680新增完整局/18723000尝试/15GiB不改。[固定训练与完整预算](PLAN.md)。各套独立TF64与对应Target监督，不能把Long draft迁名；无结构/seed/tau搜索。每局原horizon800、1200秒/4000forward帽，所有等待、失败、准备、清理计预算。当前Goal/Spatial暂停0训练、Object超时0完整局，不声称新套件已验证。

本机CPU16核被root gem5/QEMU竞争占满，Object加载与首局均慢；实际人类提交者UNKNOWN，不杀他人/改服务。新服务器资源未知，收到后登记明确未开始工作。LIBERO90无可信同代Target，仍BLOCKED未启动。正式E2E仍须原质量门＋独占本机；并行墙钟或fixture replay不能替代。

10/22 go/no-go、10/23冻结、10/28主表、11/01 18:00中文证据包。旧4f04 paused72、所有FAILED/NO_GO/retention/22缺证、原tile与保护state/cycle/rules不改。四权限false，不自动120/独立TEST/正式性能。

[20:32完整原索引](history/ACTUAL-20261017-2032.md)逐字保留，旧文相对链接按原父目录解释；历史资源安排和准备失败不改写。
