# 唯一CUDA baseline：原两进程/六真实输入全部验收通过

2026-10-12T01:16:53.585195+08:00，Asia/Shanghai；**纯tile-sdpa，非投机采样**。

## 交付

- [唯一自有CUDA](../../../review/OMISSIONS.md)：当前安装PyTorch2.9.0+cu126/commit `0fabc3ba44823f257e70ce397d989c8de5e362c1` 所用固定FA/CUTLASS核心与参数、独立namespace编译，不调用builtin SDPA/Flash扩展fallback。
- [唯一测试入口](../../../review/OMISSIONS.md)：原逐项正确性、真实fixture oracle、self/cross eager与graph-core、full-block，容差与timing loop未修改。
- [最短构建/使用说明](../../../wam_tutorial_a100/worktrees/c0-oplab-online-norm-20260922/operator_lab/wam_transformer_tile/README.md)。必要vendor头、LICENSE、setup、原fixture/full-block replay胶水保留，不留下并行CUDA候选。

主机改动仅C++ optional默认scale和Python直接native导出，原输入检查/设备guard/stream/launch/split/kernel math不变。CPU实测旧/new scale double差1ULP，但3个FP32内核参数位完全一致；11个设备函数的41099行SASS instruction/encoding相同。**仅旧/new自有版本SASS相同，不是与Torch wheel相同**：本地nvcc12.4，wheel12.6。

## 原口径实际结果

固定两独立进程各全部六输入、7轮，GPU期间不采样管理API、不注入profiler，均exit0；每进程原288项正确性检查通过，总576。每类36960份event/同步wall原始值全部保留，完整离线审计两报告结构与原gate均通过；PyTorch self/cross backend ID1，四份长KV每进程均真实2160。

原`.02/.02`、每项median-of-round-medians CUDA/PyTorch<=1：**48/48 attention eager/core、12/12 full-block 全部通过**。下表为所有输入的中位比值，单元为r1/r2，非选出的最好轮：

| 全部fixture | self eager | self core | cross eager | cross core | full-block |
|---|---:|---:|---:|---:|---:|
| 短video Q128/K128 | 0.519231/0.509434 | 0.914081/0.914373 | 0.718750/0.730159 | 0.979143/0.979094 | 0.974797/0.970408 |
| 短action Q16/K144 | 0.565574/0.564516 | 0.962264/0.962264 | 0.569106/0.564516 | 0.977642/0.977642 | 0.953938/0.969623 |
| tile0 长video Q128/K2160 | 0.826923/0.817308 | 0.986005/0.985573 | 0.723077/0.718750 | 0.979687/0.980233 | 0.982988/0.998927 |
| tile0 长action Q16/K2160 | 0.660305/0.797753 | 0.991850/0.990964 | 0.576000/0.580645 | 0.973631/0.973631 | 0.987467/0.977871 |
| tile1 长video Q128/K2160 | 0.783784/0.817308 | 0.985566/0.986960 | 0.633540/0.723077 | 0.975723/0.980233 | 0.998940/0.998125 |
| tile1 长action Q16/K2160 | 0.806818/0.806818 | 0.995745/0.996353 | 0.580645/0.580645 | 0.975610/0.973631 | 0.977581/0.985646 |

全24个eager中位降低17.31–49.06%，全24个core降低0.36–8.59%；12个full-block中位降低0.1060–4.6062%。**只代表这些固定条件下的原门槛通过，不代表所有形状、每个单次调用或端到端任务速度。** 长video margin约0.1–1.7%，不能外推为宽裕普适加速。

[r1原raw](../../../review/OMISSIONS.md)；[r2原raw](../../../review/OMISSIONS.md)；[完整原gate离线审计](../../../wam_tutorial_a100/worktrees/c0-oplab-online-norm-20260922/outputs/tile-pytorch29-baseline-20261011-203651/host-api-align-20261012/actual-release-report-audit.json)。

## 保护与最终清理

- 原利用率<=5%、显存<=32MiB、无compute/graphics PID、既有只读打开flock、10秒确认不变，两次放行通过、第一轮在fresh预声明180秒绝对截止前；每进程600秒上限。未改变系统clock/affinity/GC、未停止外部进程/服务。controller和本任务GPU进程已退出；guest下不冒充物理宿主独占证明。
- 测试前后10组353条原保护清单无变化，207包文件冻结不变。验收通过和现存source引用/155实际include核查后，归档并移除仅tile的2退役采集脚本、8未编译旧CUTLASS头、旧build与字节码缓存。最终188文件：一owned CUDA TU、一active.so、一primary test。未改已验收CUDA/source/binary/primary test/172官方来源字节；只更新自有CUTLASS来源JSON的清理状态、加README。
- 清理后原353清单只有2个**已授权退役tile辅助**条目为空，其余351不变；99投机源码、共享模型/server、原contract/fixture loader、真实输入、旧原始证据、protected state/cycle/rules保持。没有修改投机产物或其HW。再次CPU188项/六输入元数据通过，CUDA未初始化，没有第三次GPU测试。
- [最终清理清单](../../../wam_tutorial_a100/worktrees/c0-oplab-online-norm-20260922/outputs/tile-pytorch29-baseline-20261011-203651/host-api-align-20261012/cleanup-declaration.json)；[逐字节验证归档](../../../review/OMISSIONS.md)；[清理后源保护](../../../wam_tutorial_a100/worktrees/c0-oplab-online-norm-20260922/outputs/tile-pytorch29-baseline-20261011-203651/host-api-align-20261012/post-cleanup-original-source-guard.json)；[最终资源与边界](../../../wam_tutorial_a100/worktrees/c0-oplab-online-norm-20260922/outputs/tile-pytorch29-baseline-20261011-203651/host-api-align-20261012/final-resource-and-boundary-audit.json)；[CPU清理后核验](../../../wam_tutorial_a100/worktrees/c0-oplab-online-norm-20260922/outputs/tile-pytorch29-baseline-20261011-203651/host-api-align-20261012/post-cleanup-CPU-source-checks.json)。历史输出中旧路径仍是当时记录，不作为当前live依赖。

## 失败记录与因果界限

上一版两次测试预算用完，r2长video full-block慢0.4013%（7.67994us），原全gate失败，**报告保留，不因本版通过撤销**。[上一版完整结果](PYTORCH29_RELEASE_FIRST_20261012.md)。旧2.6长KV+2.7763%、低扰动+13.5854%/+1.7612%、Nsight/marker扰动、CPU计时/invariantTSC/迁移/记录成本失败均保留，不降低或取消门槛。

本轮还保留CPU辅助脚本的全部失败：GPU前首次freeze错误将预存构建/bytecode缓存计入197 source清单断言；收尾脚本首次遗漏旧build/lib目录；审计复制后首次缺parent helper导入路径。已按原归档核对10份预存额外文件并冻结207、修复辅助路径/清单口径；不改算子或原测试。GPU后边界辅助首次把无序source map的两个合法退役条目作顺序比较，随后HW录入过早等待缺失的资源审计文件，两次失败也保留；改为两个确切路径的无序比较后逐项核验，未删保护项/改门槛/补GPU。原CPU clock校准仍失败，本任务不续接不可信计时探针。

r1长video的绝对full-block中位3.160ms、r2约1.909ms，共享主机提交波动仍可见；全部原轮保留且原每报告比值通过。**主机接口确有实质源码差异、去掉Pythonshape/pow/转发，但不证明它唯一解释全部历史回退或7.68us。** 无profile根因定位完成承诺。

## 收尾与下一步

此次有限baseline源码/性能复现及唯一代码/测试交付要求已完成。用户先阅读这份baseline，之后再协商优化方向及新实验预算；**不自动追加GPU测量、episode、训练或进入fusion/persistent等研究**。旧M01–M04状态不用于此次tile验收，也未改protected状态。

[统一最终证据](../../../review/OMISSIONS.md)；[完整动态最终摘要](../../../wam_tutorial_a100/worktrees/c0-oplab-online-norm-20260922/outputs/tile-pytorch29-baseline-20261011-203651/host-api-align-20261012/final-acceptance-summary.json)。


### 最终辅助校验口径更正（2026-10-12T01:23:59.555975+08:00，Asia/Shanghai）

最后CPU完成审计初次错误期望同一lifecycle ID在log文本出现2次（该entry并没有重复record字段）；实际YAML中恰好一条ID及一份合法Knowledge记录。失败审计原文件保留，按**唯一ID条目=1**语义核验通过，没有写/修复canonical log、删历史或重测GPU。其余15项完成核验第一次已通过，source/binary/test及两实际报告SHA再次不变。最终16项全过记录见[完整完成审计](../../../wam_tutorial_a100/worktrees/c0-oplab-online-norm-20260922/outputs/tile-pytorch29-baseline-20261011-203651/host-api-align-20261012/final-delivery-completion-audit-final.json)。首次旧package归档JSON的`files=197`是冻结source清单数；tar实际另有10份已核对预存cache/build，共207个常规文件，未缺源码/失败证据。[辅助差异说明](../../../wam_tutorial_a100/worktrees/c0-oplab-online-norm-20260922/outputs/tile-pytorch29-baseline-20261011-203651/host-api-align-20261012/final-completion-helper-diagnosis.json)。
