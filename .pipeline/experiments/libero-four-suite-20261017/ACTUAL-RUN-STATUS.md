# M17实际执行索引

核对：2026-10-17 17:22 北京。新三套研究已批准，不是只读计划。旧M16、4f04、FAILED、NO_GO与四false不改。

## 已完成与运行中

|作业|实际状态/源版本|原证据|
|---|---|---|
|三套可用性|固定社区revision、配套量化/norm已核；非官方，发布者SR不继承|[来源](../../../投机采样/results/runtime/p0-libero-three-suite-availability-cpu1-v1-20261017-r1/RESULT.md)|
|Object/Goal远端资产|完整20,355,686,716B，299.335s；f696293，无hash|`p0-tf-suite-assets-remote-cpu1-v1-20261017-r2/RESULT.json`，bridge远端镜像|
|Object本机回传|真实SSH stream约3MiB/s，未完整；API失败/旧SFTP部分文件保留|[import r3](../../../投机采样/results/runtime/p0-tf-suite-assets-import-cpu1-v1-20261017-r3/status.json)|
|Object完整bank|CPU7原512 BF16生产中，原3600s硬帽不改|[bank](../../../投机采样/results/runtime/p0-tf-suite-banks-cpu7-v1-20261017-r1/status.json)|
|Goal/Spatial完整bank|远端4核affinity、Torch7线程，2e47449；首准备缺scaffold失败保留，新r2沿原2h系列截止|bridge镜像`p0-tf-suite-banks-remote-cpu4-v2-20261017-r2`|
|Object pipeline|9aa66fb，controller已启动等待资产/bank，新模型/episode0|[状态](../../../投机采样/results/runtime/p0-tf-suite-object-chain-v1-20261017-r1/status.json)|
|Goal/Spatial pipeline|9aa66fb，controller30752/30753等待原Long清理/资产，未启动新model|[登记](../../../投机采样/results/runtime/p0-tf-suite-master-v1-20261017-r1/REGISTRATION.json)、bridge镜像|
|resource bridge|6baf596，凭据仅内存；Object完整回传并核对依赖后只切换本轮临时cache给Spatial，定期镜像metadata|[bridge](../../../投机采样/results/runtime/p0-tf-suite-resource-bridge-cpu1-v1-20261017-r1/status.json)|
|旧Long M16|runner28232/guard28231四worker，17:20新增171闭合局＋原69＝240，另有执行中episode|[原作业](../libero-fullscale-20261017/ACTUAL-RUN-STATUS.md)、bridge镜像|

没有新套件训练或质量结果。controller启动不等于GPU模型/完整episode已启动。原本机17:00附近hlj vLLM约16GiB及lhw进程存在，新质量按实际显存准入，不抢占/杀他人，不正式计时。

## 登记方法与资源

[完整PLAN](PLAN.md)。每套独立TF64/seed20260923，Target FIT180/CAL20（完整200局）、两批当前模型30局FIT20/CAL10；固定PARENT2000→CHILD2500→CONT3000采CURRENT，再从同CHILD训练ON3000。仅最终checkpoint，无seed/结构/阈值搜索。接收原tau/selector/bonus/树/dtype，PF2=SST。

Object全500组各臂固定本机SXM4；Goal/Spatial全500各臂分别固定AutoDL GPU0/1。最终分0–249、250–499两个固定shard，可显存许可双worker。Target前200只读复用采集原件；该部分含采集开销且与后续并行度不同，不能当正式性能基线。初态26–49另列训练未见子集，但不修改独立TEST权限。

新master起点10/17 17:11:31；10/21 16:56:31停新、17:11:31清理。三套4680新增完整局，18,723,000 target尝试，研究原件15GiB、源资产本地40GiB，等待/准备/失败/超时/清理计入生命周期。每局1200秒/4000尝试/horizon800；不足整局预留不启动；chain失败无自动重试。旧Long仍10/19 13:40/13:55边界，不借新96h重置。

全部成败、实际命令（如796）和800罚分分列；按case配对。并行计时仅探索性C_update/ΣT_round÷Σg/policy/episode成本；formal E2E UNKNOWN，须质量与无争用完整任务计时。旧fixture有效KV桶缺口UNKNOWN，没有以compact训练表冒充全状态replay。

## 下一步与估时

先确保资产/bank终态，再直接完整T_POOL200，固定训练链和最终三臂。当前新增整体粗估48–72h，96h止损；首批完整episode后按实测修订，不能把预算上限当精确ETA。10/22 go/no-go、10/23冻结、10/28主表、11/01 18:00证据包。四权限仍false；失败/基础设施中断/审计缺口全部保留，不自动开启120/TEST/新TF正式45chunk。
