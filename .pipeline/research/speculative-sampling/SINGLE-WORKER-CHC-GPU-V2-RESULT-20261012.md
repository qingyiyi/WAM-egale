# 投机采样：单worker全层C/HC新版本GPU结果与中断复盘

2026-10-12T10:42:10+08:00（Asia/Shanghai）

## 结论

**本次已正常启动真实target GPU，生产接口问题未复现；随后因其它用户进入GPU而按独占安全关停止。没有完成C/HC正确性或性能实验，不自动补跑。**

## 完成工作与阶段成果

| 关卡 | 本轮结果 |
|---|---|
| 新默认关闭GPU入口/私有监督器 | 独立revision、flags、一次性worker入口及新CPU gate绑定；服务器独立会话、SIGHUP忽略、stdio文件，与本机熄屏无关 |
| 新job CPU预检 | **159/159 PASS**：41监督器/入口/预算/资源队列回归 + 37真实生产loader正负例 + 81既有真实接口回归；同生产loader worker的30层29变化query R/C/HC CPU probe亦PASS |
| 空闲资源确认 | 15个稳定空闲样本，无GPU进程后于10:33:24启动；忙时每180秒等待授权已实现，本次启动时无需排队 |
| 真实GPU前置准入 | **PASS**：实际trained target通过生产`modules.model`加载；全30层、固定源码、原forward/原SDPA，未patch，检查时0 target前向。以前模块别名误拒没有再出现 |
| 恢复/捕获 | **中断，不是PASS**；日志已出现reset/A_off预加载、视频21/21进度，未收到完整29-query捕获结果 |
| C/HC数值、十二类失效、13行隔离、显存 | **NOT_RUN** |
| 六块干净OFF/独立ON | **NOT_RUN**，无新速度或瓶颈占比结果 |
| 清理 | **PASS**；本任务worker/两个controller均退出，所属端口释放；其它用户进程未信号，复盘时仍存活 |

GPU加载至清理 **57.032秒（约0.95分钟）**，远小于45分钟；单worker一次attempt，0完整episode/EGL/完整policy chunk/环境动作/训练。`Infer One Chunk`日志表示开始推理，不是完成chunk。

## 为何停止，是否又是监控误报

- 10:34:20的NVML样本同时出现本任务PID1005303和新PID1009191，后者不属于本任务进程树。
- 只读`/proc`身份核查确认后者是用户 **hlj / UID1005** 的独立benchmark，不是本任务子进程；监控**不是误报**。
- 共享显卡轮询只能发现“此刻空闲”，不能阻止别人随后启动。独占性能验证不能拿混跑样本出结论，因此按授权停止；未杀对方、未放宽门、不继续消耗预算。
- 当前监督器已退出，不会在下一次空闲时偷偷重启；“每3分钟自动启动”授权已经用于本次一次性尝试，失败停规则继续有效。

## 原始账本的0必须澄清

原`FAILED.json/status.json/BRIEF.md`记录的“预留52、尝试0、完成0”是**父监督器最后收到的阶段确认**。worker在整个capture命令完成后才写回账本，本次在写回前被SIGTERM终止；实际target数学已开始，**不能把这两个0当最终实际次数**。

精确尝试/完成次数均未知，未落盘，也没有补跑来猜数。通过已冻结源码与唯一capture命令核查，RAM中的before-forward硬门限制本关**最多52次target前向**（非假定恰好52次），未进入任何后续关。900全局硬限未突破。原始FAIL/brief不改，用本补充报告和新HW记录澄清。

这暴露一个独立的**中断观测缺口**：下一轮建议只CPU补目标前向attempt/completion的中断持久化与真实worker终止回归，保持数学/attention/计时边界不变；如果持久化有成本，应一致收费并隔离OFF计时，不隐去开销。本轮只诊断，没有修改已执行源码或重试。

## 不变与证据边界

只读CPU核查：298份前置文件 + 17份本次私有实现 + 49份raw job文件及既有HW原始record保持SHA；9份target权重stat、A_off及fixture不变，受保护state/rules不改、cycle仍不存在。此前6份CPU数学/准入文件与新版本逐字一致。自审非独立审计。

本轮只推进了正常启动与真实生产接口准入，没有证明完整verifier加速、policy成功率、环境步数或E2E。仅投机采样；TILE既有记录增量保留，不混写其成果。

## 下一步、目标

建议下一步先只CPU补中断时账本持久化及真实worker终止回归，同时协调可靠的45分钟独占GPU窗口。CPU通过后另确认新job：仍单worker、最多900次target前向、45分钟含加载清理、0episode/EGL/完整chunk/训练；恢复/捕获→数值/失效/隔离→六块OFF→独立ON，失败停。当前FAIL不复用、不补跑；本轮没有自动下一实验。

目标仍是：先证明长历史单worker的全层历史KV复用不扰动真实输出/状态且显存可承受，再测含冷构建/staging/检查/释放的完整verifier净成本；有净收益后才另规划完整policy/冻结target闭环。不重构方案、不自动mega/训练。

[CPU预检](../../../投机采样/results/runtime/p0-single-worker-chc-v2-20261012-102720/preflight/CPU-GATE.json)；[真实GPU准入](../../../投机采样/results/runtime/p0-single-worker-chc-v2-20261012-102720/worker/INTERFACE-ADMISSION.json)；[原始失败](../../../投机采样/results/runtime/p0-single-worker-chc-v2-20261012-102720/FAILED.json)；[清理](../../../投机采样/results/runtime/p0-single-worker-chc-v2-20261012-102720/OWNERSHIP-CLEANUP.json)；[只读复盘核查](../../../review/OMISSIONS.md)；[原始简报（计数0见本报告澄清）](../../../投机采样/results/runtime/p0-single-worker-chc-v2-20261012-102720/BRIEF.md)。
