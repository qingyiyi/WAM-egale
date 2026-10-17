# 执行接线约定

本目录为获准的新源码namespace，固定数学源4f04b35；不修改旧目录/公开快照。计划路径为CLI --job/job，读取job/plan.json，无hash。用户AutoDL无smoke/独立测试/SHA，必要审计跟正式完整episode。

- protocol.py: ROOT/EXP/WORKTREE；read(path), write(path,obj), require(condition,message), load(job)返回全新计划且不hash；episode_dir(job,pair_id,arm)由runner明确传CLI --out；source不自动启动GPU。
- server_entry.py实际CLI：--job --worker-index 0|1 --port --master-port [--forward-cap 24500]。每卡一个持久server，从plan读取selected TF_ON及相同完整reset bank，通过deployment.py无SHA入口加载；写READY.json或FAILED.json。各episode的pair/arm/out由真实reset请求的p0_pf2_context传入，不以进程CLI固定arm。
- client_entry.py实际CLI：--job --pair-id --arm C|P --out --port --worker-index [--master-port]。直接完整LIBERO horizon800，写EPISODE-TERMINAL.json、commands/RNG/初始原件。无hash；无独立测试。初态与first2S保存/对照由server/client协调，不注入对方状态。
- report.py: --job，从job/episodes/pair-XX/C或P读取终态/计数/审计，输出已登记CSV/报告和PANEL-STATUS.json；允许PARTIAL，不补失败/不签四false。
- deployment.py: load_predictor(plan, server)绑定原ActionPredictor预测数学，避免旧load/sha；write/read直接元数据。具体factory签名先短消息同步server作者。
- state.py: 无hash比较/捕获RNG、全部有效KV、原始物理/图像元数据；first2S使用server持久RAM且同pair共用worker，严禁每arm重新server导致first2S RAM丢失。若需跨进程则用受cap小共享CPU store，不能逐局写30层完整KV。

关键：每卡一个持久server，同pair两臂通过episode reset切换C/P schedule，先后顺序按plan；不得同卡两个target模型。已实现的p0_pf2_context元数据路由保持原reset业务参数/初始化/RNG；C和P只改变周期参数。保留独立S/T审计schema，原ObservedAPI不改。完整KV仅在同pair现场RAM比较，持久化收据不能冒充离线全KV重验。

本文件为运行结束后的文档修正；实际六对DEV执行源码身份仍为9acc37cee9ae9b62b8311f1f8dbb01bffa37a86a，不将后续文档提交冒充已运行版本。

根任务负责runner.py、远端部署/guard与全局预算；子任务文件所有权见各自指派。协议接口调整立即消息同步，必要时更新此文档，避免不同agent同时写同一文件。
