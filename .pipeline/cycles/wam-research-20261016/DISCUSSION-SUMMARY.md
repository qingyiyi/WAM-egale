---
kind: discussion-summary
cycle: wam-research-20261016
updated: '2026-10-19T13:37:18.769422+08:00'
---
# 当前讨论与授权

用户同意先做本轮1/2再汇报并结合源码讲解。M26已完成强基线/profile和一个FFN子图的有限原语验证：完整device FFN2不合格，保留失败；收缩FFN1+GELU采用原BF16中间舍入、FFN2继续Torch，六个真实完整block两个进程逐位与状态过。B14仅2.45–3.23%通用block延迟下降，不扩为候选DAG或MegaTransformer创新。下一步先讨论FFN2原语契约和完整block可调度边界，未启动persistent/P_seq/P_sched。

用户要求只维护baseline/optimized两份活动实现，代码简明，投机侧禁止改动。新设备/静态入口均在算子研究/optimized，共享tools可完善；默认install_model未自动切新FFN，旧16目录308源留Git历史。原tile/shared-prefix仍是投机依赖，原位只读。代码整理和文档不改变当前质量/数值标准。

本机按自然空闲单GPU/CPU≤2，用户不再要求反复协调。M26全部本机GPU作业已终态、自有退出；未用预算不自动接续。NCU初轮默认时钟控制漏关闭已记偏差，实际改频未知，未来命令强制none，不把counter当正式性能。原空卡NVML异常仍UNKNOWN。

远端用户要求两卡各并行、后继自动后聚焦本机算子。已完成运维交接，固定分片最多每卡2worker；匹配T200同卡限制使当前3活动，后继自动排队，不重跑已完Target充占用。M17原起点和10/21截止不变，CPU只读watch每5分钟；BAAI一切写入仅~/cxz。AutoDL不再SSH，关机未知。新机Codex已装不等于会话迁移。

engineering_quality_screen=false、independent_test_complete=false、noninferiority_established=false、formal_performance_permission=false。旧FAILED/retention/NO_GO/paused72/96、MLP0、22缺收据/default A_off不改。TF64/TF_ON、normalizer、tau、K2B2/bonus/prefix_then_cost、BF16/FP32语义冻结。PF2=SST完整policy，T50更新/51action调用含末尾；reset/cache不计周期。各臂同用512-token BF16优化reset，cold另列；actual796与800罚分分列。

投机源码/配置/运行依赖和baseline不改；本轮1875投机文件复核不变。原tile/shared-prefix依赖原位只读，保护state/cycle/rules不覆盖，不建缺失cycle.yaml。不自动新训练、120/独立TEST、新TF正式45chunk或第二环境。BAAI写入永远仅~/cxz，不杀他人/改服务/手动时钟。AutoDL不再SSH，关机UNKNOWN。GitHub仅main镜像；中文论文会话交飞书，不建初稿。

10/22 go/no-go、10/23冻结、10/28主表、11/01 18:00中文证据包。完整早前讨论见[本次历史](history/M26-before-result-20261019/DISCUSSION-SUMMARY.md)。
