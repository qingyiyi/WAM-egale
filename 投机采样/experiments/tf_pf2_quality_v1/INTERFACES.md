# 执行接线约定

本目录为获准的新源码namespace，固定数学源4f04b35；不修改旧目录/公开快照。计划路径为CLI --job/job，读取job/plan.json，无hash。用户AutoDL无smoke/独立测试/SHA，必要审计跟正式完整episode。

- protocol.py: ROOT/EXP/WORKTREE；read(path), write(path,obj), require(condition,message), load(job)返回全新计划且不hash；episode_dir(job,pair_id,arm)由runner明确传CLI --out；source不自动启动GPU。
- server_entry.py: --job --pair-id --arm C|P --out --port；从plan读取selected TF_ON，通过deployment.py no-hash loader；写READY.json或FAILED.json；本地既有T/MLP arms不导入新臂身份。server只监听当前worker端口。
- client_entry.py: --job --pair-id --arm C|P --out --port；直接完整LIBERO horizon800，写EPISODE-TERMINAL.json、commands/RNG/初始原件。无hash；无独立测试。初态与first2S保存/对照由server/client协调，不注入对方状态。
- report.py: --job，从job/episodes/pair-XX/C或P读取终态/计数/审计，输出已登记CSV/报告和PANEL-STATUS.json；允许PARTIAL，不补失败/不签四false。
- deployment.py: load_predictor(plan, server)绑定原ActionPredictor预测数学，避免旧load/sha；write/read直接元数据。具体factory签名先短消息同步server作者。
- state.py: 无hash比较/捕获RNG、全部有效KV、原始物理/图像元数据；first2S使用server持久RAM且同pair共用worker，严禁每arm重新server导致first2S RAM丢失。若需跨进程则用受cap小共享CPU store，不能逐局写30层完整KV。

关键：每卡一个持久server，同pair两臂通过episode reset切换C/P schedule，先后顺序按plan；不得同时两个target模型。server episode arm不能伪造checkpointcondition。建议server --worker-index 加 /begin-episode RPC或原reset request中的独立p0_pf2_context字段，以实际arm设periodic，保持业务原reset请求其他字段/初始化/RNG不变。必须保留新schema与两臂相同wrapper，不改变ObservedAPI。

根任务负责runner.py、远端部署/guard与全局预算；子任务文件所有权见各自指派。协议接口调整立即消息同步，必要时更新此文档，避免不同agent同时写同一文件。
