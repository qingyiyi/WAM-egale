# Long阶段189组三臂与可暂停接续：2026-10-17 22:08北京

本文件是新授权工作的当前检查点，不覆盖4f04 paused72、b0科学基线或review/RESULTS.json历史身份。源实现固定2b1f0395664dea629a109718d6657128495d7db0，阶段统计源7ca036a45d66a58ad715e71911e72e516c756c3f；旧数学来源d5448bbc5b02465dcc8d7c6df5bb4497d40cf3d9。

旧AutoDL按用户要求阶段暂停，实际189组三臂/567局全部自然终态、0partial，311组未开始。所有37349个runtime/配置/小元数据文件、固定TF与normalizer、全部reset bank和失败证据已备份到本机：源1970600995B，tar2053969920B，650.7秒，逐entry/字节/stat核对，无SHA、无远端源删除。原v1备份SIGINT130/FAILED及旧supervisor PARTIAL保留。原始tar/模型/相机/NPY/KV不公开，不声称仅凭本仓库已重算原张量。

|方法|成功/完整局|平均800失败罚分|与Target共同成功最大长度比|
|---|---|---|---|
|Target|183/189|284.81|1|
|TF连续|185/189|274.76|1.7004|
|同TF＋PF2|181/189|284.22|1.5465|

所有成败均纳入。旧C/P组件审计FAILED分别15/27保留，包含训练暴露初态；这些数值不是独立TEST、质量通过或TF优于MLP的证据。共享并行墙钟及ΣT_round/Σg仅作探索成本，不签正式E2E。四项engineering_quality_screen、independent_test_complete、noninferiority_established、formal_performance_permission始终false；旧NO_GO、FAILED、retention及全局A_off不改。

用户最新允许新服务器接入前继续有用工作。已在新namespace启动只接续311未开始case的四模型作业（四模型已全部READY、首30局分片运行中；两卡利用率约98%/97%）；每10组三臂/30局形成可暂停屏障，不重跑旧567、不重启已终态旧job。所有旧attempt累计debit591787，沿原10/17 13:55起点、10/19 13:40停新/13:55清理、6001000forward/10GiB限额。全部臂仍完整512-token BF16优化reset，无smoke/独立测试/SHA，无新训练/阈值变更。

新机SSH未收到，未迁移；新job后续结果不在上述冻结归档内，换机前需增量备份、真实清理，当前不可关闭旧实例。Goal/Spatial原job暂停0训练，本机Object首局1200秒超时/CLEAN、0完整局，不自动重试。LIBERO90仍缺可信同代Target、未启动。

证据：[阶段结果与逐局表](../.pipeline/experiments/autodl-stage-migration-20261017/STAGE189-RESULT.md)、[当前Cycle](../.pipeline/cycles/wam-research-20261016/PROGRESS.md)、[真实多代接续源码](../投机采样/experiments/tf_fullsuite_wave_resume_v1/runner.py)。新作业实际状态继续看HW，不能把本文快照当实时进度。
