# WAM项目短入口

更新：2026-10-19T13:37:18.769422+08:00。当前Cycle：[wam-research-20261016](cycles/wam-research-20261016/PROGRESS.md)。先约束记忆，再读当前四文；旧大索引/原始trace按需。

## 当前结果

- **本机M26已终态**：[强基线与FFN原语全表](experiments/operator-profile-device-20261019/RESULT.md)。六真实B2/B14×短中长block0，30eager/24Graph组合全部逐位/有效状态过，实际GEMM M32→224、每候选attention仍Q16。强cache路径FFN占trace kernel时间21.9–29.9%。
- **完整device FFN未过**：FFN1/GELU六点逐位过，FFN2六点失败，四个固定分块公式均未复现；根因UNKNOWN，未降容差。最终收缩FFN1+GELU保留原BF16舍入、FFN2仍Torch，六点两独立进程中间量/完整block/有效KV/metadata/输入/RNG/参数字节过；B2延迟降0.74–1.52%、B14 2.45–3.23%。这是通用block小优化，不是新MegaTransformer调度/30层/chunk/E2E。
- **本机资源**：17作业终态，751.141秒生命周期、1093 block host/capture/replay intents、617原语；两工具失败保留、无timeout，自有GPU退出。NCU漏显式关闭默认boost的配置偏差保留，实际改频未知；后续命令强制none。profile不进性能表，空卡NVML24%异常仍在。
- **两份活动代码**：[算子研究](../算子研究/README.md)。baseline与1875投机源码/配置不改；原模型固定ca5f37e字节相同、block forward AST同原。新静态/FFN入口只在optimized；默认安装不自动启新FFN。旧16目录308源码留Git a0ed2d8，原tile/投机shared-prefix依赖原位保留。
- **原M25证据**：[两进程108检查与432full30](experiments/operator-stable-timing-20261019/RESULT.md)。有限逐位过，最大完整模型中位回归2.18%，严格±2.5%非全点；旧三轮争用FAILED/PARTIAL不改。B14 full30真fixture，B2 full30为root CFG0/7派生，不是顺序轨迹。
- **远端Object**：13:31 T356/C156/P156，668/1500，3worker活动、后继自动排队、无terminal；最多两卡各2，同卡Target配对限制不绕过。每5分钟CPU巡检继续，原M17预算/截止不变。[运维](experiments/remote-eval-parallel-20261019/RESULT.md)、[本轮快照](experiments/operator-profile-device-20261019/REMOTE-STATUS-READONLY.json)。
- **完整质量覆盖**：Long T483/C484/P473；Goal463/464/467；Spatial472/466/464，各500。Long罚分均值285.858/282.588/293.270，双方成功仍有长尾；各套首两S严格配对FAILED/训练DEV曝光不改。Object固定14补采/准入/ON已完，旧无效监督/失败保留。正式E2E UNKNOWN。

## 当前下一步

本轮报告/源码导航已完成，main发布与回传在收尾；后续明确完整block DAG、FFN2契约与可调度边界。P_seq/P_sched未启动，约3%融合不能代替调度贡献。g/ΣT_round÷Σg/完整chunk和episode收益本轮未测。主线仍验证“投机重组负载→合法候选并行→新增执行组织→质量约束下有效成本”，不是把普通cache/batch改名。

## 必留边界和日期

engineering_quality_screen=false；independent_test_complete=false；noninferiority_established=false；formal_performance_permission=false。旧NO_GO/FAILED/retention、paused72/96、MLP0、22缺收据/default A_off不改；TF64/TF_ON、normalizer、tau、K2B2/bonus/prefix_then_cost、BF16/FP32冻结。PF2=SST完整policy，T50更新/51action调用含末尾，非bonus或状态恢复；所有臂用512-token BF16等价reset，cold另列，actual796/800罚分分列。

M17原10/17 17:11:31起、10/21 16:56:31停新/17:11:31清理、18723000尝试/15GiB，补采不重置。M26 10/19 12:51起/10/20 00:51清理，单worker/CPU≤2、7200秒GPU/6000block/20000原语/1GiB，本轮已终态不自动花余额。旧预算不复活；不自动重试/训练/120/独立TEST/被阻塞新TF正式45chunk。BAAI所有写入仅~/cxz；不杀他人/改服务/手动时钟；AutoDL不再SSH、关机UNKNOWN。保护state/cycle/rules和原tile HW不覆盖，不跑旧全量sync。

GitHub仅main进度镜像，固定旧科研commit不被替代。中文论文会话给用户整理飞书，不建初稿。10/22质量/系统go-no-go、10/23冻结、10/28主表、11/01 18:00中文证据包；投稿官方节点按既有独立核验记录，不虚构abstract期限。

详细旧进度在Cycle history及各实验原件，默认不展开全部旧知识库。
