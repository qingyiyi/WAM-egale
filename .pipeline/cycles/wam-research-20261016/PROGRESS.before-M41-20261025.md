---
kind: progress
cycle: wam-research-20261016
status: active
updated: '2026-10-25T01:02:58.552986+08:00'
plan: PLAN.md
current: M39
next: M39
---
# 当前进度

M39-SM120-FFN-NEXT已闭合：T0数值/旧负结果、T1成本、唯一PAIR2048x2实际完成。三fixture七段P/R与原捕获中间量/输出、输入/KV/RNG、新384-owner schema有限逐位通过，但FFN慢约18%–19%、正常整层两进程三KV慢6.25%–7.89%，NO_GO，保留当前A且原入口未改。正常资源完全相同（170CTA/shared100352/reg255/staticlocal784），不能归因为shared预留增大。完整17阶段表已按T0/T1/T2输出并保存，T3复用同轮最新值。5214完整block、作业生命周期452.531秒，CUDA active未知；13phase与5guard身份退出，5090 compute context为空，实例未关。原科学门全部false。见[本轮结果](../../experiments/sm120-ffn-next-20261024/REVIEW_RETURN.md)。

H02本机轻量化执行中：用户明确A100跑已授权质量、5090测性能、本机保留开发/轻量证据与临时任务，已Maintain。bashrc及Codex子进程TMPDIR/TMP/TEMP统一/rjs/cxz/tmp，新login shell/Python临时文件位置已核验。闲置/tmp623项已私有归档并清除、1783文件逐字节复核，28活动/近期项保留；该批释放/tmp约80.71MiB，私有压缩归档约41.23MiB。Long/RoboTwin公共text encoder与VAE七文件cmp一致，原子hardlink保留路径/字节，释放13.2068GiB。大文件25564件冷归档正在向现有BAAI ~/cxz/backups独立目录传输，计划120.869GiB、预期本地再回收120.619GiB；按用户最新要求改为逐文件传输校验、接收端落盘确认后立即删除该源；每30秒更新已释放数、远端尺寸收据与按inode核算的实际回收，不等整批。原第一轮因远端缺rsync退出12，0数据删除；失败收据保留，项目内便携rsync已配置，旧无删除传输为切换新策略主动停止（退出255）；原断点、全清单、各attempt终态保留。当前逐文件runner沿22:21:19原起点累计12小时，最迟10/25 10:21:19清理。当前传输不代表归档完成。

[存储整理入口](../../maintenance/storage-placement-20261024/PLAN.md)、BULK-STATUS.json及恢复脚本同目录。原tile、投机源码、科学四false和A100队列未修改。

M39上轮FFN四方案已实际完成并止损：B专职TMA输入、C跨任务MMA/epilogue交接、D单CTA ACT驻留、E双CTA cluster/DSM共享。四者三KV真实B14/Q16/block0的27捕获中间量/输出、输入/全部有效KV、CPU/CUDA RNG及各自提交schema均有限逐位通过，但两个独立进程无插桩完整层均慢于冻结A，不采用。B整层增加1.201%/1.129%/0.833%，C增加0.900%/0.603%/0.722%；D为A的2.256/2.146/2.033倍，E为3.123/2.932/2.745倍。

最后同轮交错profile对照确认FFN局部也无收益：B耗时约1.028/1.034/1.030倍A，C约1.015/1.013/1.008倍，D约4倍、E约6倍。保留SELF-QKV-TMA-4MMA，实际优化源码/DSO本地与5090六文件逐字节不变；新版当前A完整17阶段表已新测并在对话打印。当前A无插桩原Torch/Own为1.2514/1.7546/2.1902×，Guarded/Own为1.1127/1.0845/1.0453×；共同warm CrossKV、eager/no Graph、单层非机器人E2E。

ACT驻留确实避免正常FFN2从全局ACT读入，无FFN1重复计算；但M16使FFN1权重tile逻辑请求180→1176MiB、FFN2 336→1176MiB，非HBM实测。D仅98个驻留任务工作；E实际85个双CTA cluster、DSM读取与前后cluster同步均可运行，不能再断言5090不支持该实际路径，但本映射明显无收益。没有NCU/实际重叠时长及各因素定量分离；不把容量或源码并发能力当实测cache-hit/重叠率。D首次模板编译失败保留，r1直接warp MMA后有限通过。

上轮5个编译/实验job加一次同轮归因，生命周期之和1207.323秒（编译1095.558秒，GPU工作进程生命周期108.073秒，CUDA active未知），13380完整block；1次编译失败、0 timeout、4个性能拒绝。33个自有PID+出生身份及阶段进程组已退出，5090 compute context为空，实例未关机；远端新增约93.3MiB。用户新现有5090累计不限额授权适用，旧账本/失败/截止不改，各job仍有界。双A100预算不扩。

M40最新2026-10-25T01:01:57.546414+08:00实查405/600完整三臂组、1215/1800局，Target/TF/PF2成功数326/328/332（同分母405）。00:53 pair406/T观察VAE CUDA OOM触发HALT，旧记录及C/P自然失败保留；01:01新GPU1 namespace加入加载前24GiB/开局前6GiB显存准入，只恢复未开始组，两卡当前均运行、无HALT。pair406/T不自动重试；原pair281/C已授权尾部恢复保留。已用约86.05/160 GPU小时，原T0/更早guard截止不改。见[新中断与接续](../../experiments/robotwin-pilot-20261020/eval100-20261022/external-oom-20261025/PLAN.md)。

M40此前快照：双A100持续原RoboTwin固定队列。17:39:56巡检335/600有效完整三臂组（1005/1800局）、2组执行、262未领取，两worker均RUNNING_COMPLETE_DEV_EPISODE且无新HALT。原STATUS竞态与pair281 C中断保留，尾部一次恢复按原方案审计；不以worker退出代替600有效组。原160GPU小时/10天不变。

下一步建议保留当前大M与共享全局ACT，先独立归因FFN1写回/尾部与K片段ready，再决定新候选，不继续当前M16完整ACT驻留映射；本轮未自动展开新实现。未知在线接受/其他层/完整模型/机器人质量，engineering_quality_screen、independent_test_complete、noninferiority_established、formal_performance_permission与production_admitted均false。baseline/投机/protected文件未修改。

[本轮结果](../../experiments/rtx5090-block-port-20261023/ffn-local-handoff-20261024/REVIEW_RETURN.md)；[当前完整阶段表](../../experiments/rtx5090-block-port-20261023/ffn-local-handoff-20261024/STAGE-TABLE.md)；[固定源/结果入口](../../experiments/rtx5090-block-port-20261023/ffn-local-handoff-20261024/SOURCE-AND-RESULT-ENTRY.json)。

[既有Self QKV采用与公开两流证据](../../experiments/rtx5090-block-port-20261023/projection-warp-specialized/self-qkv-tma-migration/REVIEW_RETURN.md)；[双卡接续与失败恢复](../../experiments/robotwin-pilot-20261020/eval100-20261022/resume-after-storage-race-20261024/PLAN.md)。
