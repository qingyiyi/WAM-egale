# 三小投影Split-K测量入口

正常候选：`wam_whole_projection_splitk_cuda`；诊断候选：`wam_whole_projection_splitk_profile_cuda`；冻结A：本目录 `A.so`，真实模块名 `wam_whole_splitk_cuda`（已包含采用的FFN2 Split-K）。不修改历史FFN2实验或Torch。

先使用 `precision_unique.py` 验证三真实B14/Q16/block0 fixture，再用 `measure.py` 两独立进程四臂比较：原Torch／Guarded／A／本候选，全部eager、相同Cross KV缓存。仅进程1传 `--stages`。输出名沿 `results/PROJECTION-measure1-data` 和 `…2-data`，`--label PROJECTION`；候选binary路径与module名显式传入。原Torch独立阶段和插桩诊断不代替无插桩完整层。

FFN2计数门保持：96个输出tile各7段，四组领取168、完成24。新增投影计数：`projection_splitk_commit=[3,96]`全部10，`projection_splitk_claimed=[3]`全部960；顺序为Self输出／Cross Q／Cross输出。缺字段或数值/状态/计数FAIL均返回非零。

正常核心源码冻结后运行本目录 `make_profile.py`，只生成本目录诊断源与build脚本。profile入口保持原17区间，不新增同步。报告 `report.py PROJECTION --retained '实际当前版本'` 保存并打印完整表。三投影各960计算任务、8W MMA、M64N128、九段K320＋一段K192；保留原两处grid同步。Self QKV和已采用FFN2不改。

最终public入口用 `precision_public.py`：真实调用 `whole_api.extension()`、核模块名和binary完整路径，分别 `--stream default` / `custom`。默认要求FFN2及新增投影门。若回退保留仅FFN2-SPLITK，则显式传 `--expect-projection-splitk=false`，仍保留默认 `--expect-splitk=true`；不可用关闭门的参数核验投影候选。

所有脚本依赖本目录真实 `JOB.json` 截止，保持每测量进程6500完整block硬限与根执行器登记的总预算。未在本CPU准备步骤运行GPU、SSH、SHA或smoke。源脚本已AST核验；GPU结果尚未取得。
