# 最近执行检查点

## 2026-10-17 20:32 北京：三卡实时核对，本机首局超时已停

AutoDL GPU0/1分别78%/83%、约28GiB，Long各2worker继续；20:31新423闭合完整局＋旧69＝492/1500（164三臂组），另有进行中局。GPU0当时worker0 Target/worker2 PF2，GPU1 worker1 Target/worker3连续TF；路由随预登记case交错，不是按方法固定GPU。Goal/Spatial仍WAITING、active_steps空，Spatial还缺完整Transformer资产；轮换/归档继续等Long全量与实际清理。

本机Object实际READY，setup749.6秒；首task0/init0于20:04开局，20:23:58达到1200秒上限，PARTIAL_HARD_FAILURE_NO_RETRY，完整episode0，不记自然任务失败/800罚分。20:24campaign终态FAILED_OR_PARTIAL_NO_RETRY、guard CLEAN/owned组空，GPU显存0。CPU0–15本次1秒采样全部100%；CPU争用明显，但未单独profile证明全部耗时根因。不重试、不改原预算/终态。证据：`p0-gpu-status-readonly-20261017-2030-r1/{REMOTE-STATUS,GPU-STATUS-SUMMARY}.json`与原Object v4终态；只读SSH已关闭，无新GPU作业。

## 2026-10-17 19:51 北京：空闲本机分担Object，完整链已启动

用户新授权已用Maintain记录。nvidia-smi无可见compute进程/显存0MiB，利用率约25%不签独占；CPU0–15忙，故仅本机一个model、nice10、server8/9/client10/11、control7，内存24GiB/显存22GiB阶段准入。source652e83cf96a9af0585801a093589becc877cad9b，独立`p0-tf-suite-object-local-chain-v4-20261017-r1`，controller518248/birth1350697010/UID1003已实际启动完整T_POOL。尚未声称READY或episode完成。

CPU1导入已完成原512 BF16 bank128.9秒，只改checkpoint_path元数据，T5/teacher调用0，子进程exit0/SSH关闭；`p0-tf-suite-object-bank-import-local-cpu1-v1-20261017-r1/RESULT.json`。远端零工作Object32543/birth6696082227经原状态/UID/PGID核对退役，原FAILED_OR_PARTIAL_NO_RETRY保留，新receipt记录是资源替换。Goal/Spatial/Long、轮换与归档不动。

[v4登记](../../../投机采样/results/runtime/p0-tf-suite-local-placement-v4-20261017-r1/AMENDMENT.json)保持M17原17:11起点、10/21停新/清理、1560局/6241000尝试/5GiB；最终两250case shard顺序，同case全部臂固定本机SXM4，禁止混PCIe计时。旧4false/FAILED/tile/保护不变。CPU忙可能放慢装载，900秒原setup帽不改。

## 2026-10-17 18:48 北京：归档已登记启动等待

128d435独立CPU1归档producer已实际启动，本机443492/birth1350318267，`p0-long-evidence-archive-local-cpu1-v1-20261017-r1`，状态WAITING_FULL1500_AND_STRICT_OWNED_CLEANUP；复制0文件/0B。保持17:11/master96h截止，完整manifest后传输≤10800秒，总≤12GiB、单文件64MiB、metadata128MiB、本机实际空闲≥5GiB；密码仅RAM。先等Long1500及原own实际清理，再等37f空间轮换完成审计，全部原件复制、INDEX/COPY-RESULT持久化后才退远端重复bulk和精确COUNTER；其他JSON/FAILED/proof/plan/md/csv/log均保留。旧源不改，无GPU/test/SHA，无自动重试。登记在`p0-tf-suite-storage-placement-v3-20261017-r1/ARCHIVE-REGISTRATION.json`。

## 2026-10-17 18:42 北京：已有磁盘分配与归档准备

Object准备仍零模型/局，现独立OS盘job `/root/wam-suite-research-20261017-v3/object-chain-r1`、controller32543、source8eaf130；原R2只补替换原因并退出。5GiB套件上限及5GiB实际空闲线不变，Goal/Spatial在数据盘。同17:11起点/10/21截止；归档producer准备中，尚未复制或清除Long结果。

18:42现场Long新285闭合局＋旧69＝354；三套完整bank已齐，remote新训练0。37f15d1轮换调度32254仍等待全Long及清理，尚未删模型。原本机完整权重保留；项目只读du约144.52GiB，本机余94.9GiB。491个保护源size/mtime全部一致（不计算hash）。最新真实[空间/状态快照](../../../投机采样/results/runtime/p0-tf-suite-cache-rotation-inputs-cpu1-v2-20261017-r1/ACTUAL-CAPACITY-AND-PLACEMENT-r1.json)。

## 2026-10-17 18:10 北京：本机减负、暂不扩容

用户最新选择已通过Maintain保存：新增重计算优先现有AutoDL，先清理、不扩盘。本机7个已关闭失败下载残片985,044,916B已清理，清单在`p0-tf-suite-abandoned-download-cleanup-cpu1-v1-20261017-r1`。原失败/完整权重/原始结果保留，本机剩约94.9GiB。

新source8eaf130沿用9aa数学/训练/阶段预算。Object31851/Goal31852/Spatial31853远端控制器已启动等待；原17:11起点和10/21截止不改。GPU0前两套、GPU1 Spatial，同卡原子双槽、CPU租约、实际清理后释放。旧准备控制器零模型/局退出，原记录不改。当前仍等待Long、Object bank、Spatial资产，新增训练和机器人局0。

Goal/Spatial完整bank成功（CPU2455.0秒）；Object远端v3 r3生产中，缺原bank收据和启动参数错误均保留失败，补真实原件后沿原2h系列截止。本机Object资产已完整，原3200秒传输超时和旧本机bank超时保留；只有非复用下载残片另有清理。

18:05 Long新234闭合局＋原69＝303，双卡四worker继续。空间轮换要等其500case全终态及实际清理；三片远端Long Transformer缓存可由本机原件重建，仍需加载的T5与全部结果保留。轮换调度源码尚在完成，不能声称空间已释放。

## 2026-10-17 17:22 北京：原始M17登记检查点

2026-10-17 17:22 北京：M17新增三套同配方训练及全量三臂已登记。master起点17:11:31、10/21 16:56停新、17:11清理；4680新增完整局/18,723,000尝试/15GiB研究证据，原Long48h账本不重置。

source9aa66fb为完整pipeline；asset下载f696、CPU bank2e47449、resource bridge6baf596。Object本机controller等待真实权重/完整bank；Goal/Spatial远端controller30752/30753等待bank/原Long清理。尚未启动新增模型或机器人局。Object/Goal远端asset299.3s成功；本机前下载及SFTP/SSH API准备失败保留，Object真实SSH stream约3MiB/s。远端bank r1缺Spatial组件脚手架先于T5失败，r2补必要资产引用后继续，原系列2h累计deadline不改。

Long runner28232/guard28231、双卡4worker继续；17:20新增namespace171闭合局＋旧69＝240，非全量。外部本机用户hlj/lhw GPU进程17GiB级占用；新质量按显存准入，不做正式计时或影响他人。

[详细作业索引](../../experiments/libero-four-suite-20261017/ACTUAL-RUN-STATUS.md)；[全部此前执行记录](history/EXECUTION-through-M16-20261017.md)。旧四false/FAILED/NO_GO/tile/保护不动。
