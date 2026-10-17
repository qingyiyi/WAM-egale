# M18旧AutoDL阶段收尾与换机准备

用户明确要求旧实例阶段结果后暂停、准备换机。此项只改变资源调度，不取消Long/三套TF研究，不重置旧预算，不授权采购。最新用户允许新机接入前继续有用工作：先确认阶段原件备份，再新namespace按完整30局分片接续未开始组。Maintain决定见../../memory/global/requirements/decision-旧AutoDL阶段收尾暂停并准备换机.md。

- [x] 当前四个已进入case181/182/192/193以原d5448bb客户端和服务补完三臂，无新case、无重跑、无截断。189组/567局完整，311组未开始、partial0；原supervisor PARTIAL_OR_FAILED终态保留。四权限false。
- [x] 源fb519d2：CPU16独立有界调度，旧runner28232仅调度进程SIGSTOP，客户端/模型照常完整运行；SIGCONT回原清理。6000秒含等待/收尾/清理，metadata16MiB，无新模型/新增原forward预算；独立rescue防CPU dispatcher遗留停止。旧guard/owned组实际清理后才形成阶段清单。
- [x] Goal/Spatial等待控制器PAUSED_PARTIAL/0局、租约已释放；轮换BLOCKED/0删除/无asset child；旧全1500归档等待器源终态保留，实际RPC关闭exit0。Object之前本机超时未重试。
- [x] 源d098fd2的逐文件备份在CPU争用下过慢，已实际SIGINT退出，FAILED/CLEANUP保留、SSH闭合；改9f8f251单tar大块回传，原9000秒绝对截止不重置，旧部分副本保留。本机CPU7单核流式完整备份，不在远端创建tar或删源。所有runtime原件、TF及normalizer、完整bank、失败和小配置共约1.84GiB/37349文件。总9000秒/copy5400秒；payload4GiB，单文件64MiB，manifest32MiB，entry≤100000，本机空闲≥5GiB；密码只RAM。验证entry/字节数、源前后stat及真实退出，不做SHA、不运行smoke或独立测试。
- [ ] 本机RESULT与CLEANUP完整后复核旧机没有本项目运行进程，形成可释放确认。实例关机由用户操作，不将进程退出当已关机。
- [ ] 新服务器SSH地址/端口/用户名及GPU/CPU/磁盘配置收到后，新namespace登记接续；仅未开始完整case，可读复用旧完整组，不重复或补造原失败。同case各臂同一新物理卡，硬件/并发来源分层，含FIT/DEV不签独立TEST。

主要输入：原Long current/prior plan、完整原始终态/计数/审计、固定TF_ON、四套bank及套件元数据。大Target权重不做重复回传：本机Long/Object原件保留，其他发布revision和来源归档；新机加载前核查资产与路径绑定。

证据入口：投机采样/results/runtime/p0-autodl-migration-stage-plan-v1-20261017-r1；远端p0-autodl-migration-stage-pause-v1-20261017-r1；本机p0-autodl-migration-backup-local-cpu1-v1-20261017-r1。旧Long10/19 13:40停新/13:55清理、M17 10/21 16:56/17:11保持。10/22 go/no-go、10/23冻结、10/28主表、11/01 18:00中文证据包；证据不足收缩主张。

新机接入前继续：源2b1f039、独立p0-tf-long-shard-resume-v1-20261017-r1，已启动四模型装载。只311未开始case、完整30局屏障；原debit591787，原48h起点/6001000forward/10GiB/四false不改。收到SSH后写STOP_AFTER_SHARD，在真实完整边界收尾、增量备份、清理后迁移。新结果未备份前不释放旧机。阶段统计见[189组汇总](STAGE189-RESULT.md)。
