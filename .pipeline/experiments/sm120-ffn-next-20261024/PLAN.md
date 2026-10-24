---
kind: experiment
cycle: wam-research-20261016
status: closed_no_go
baseline: SELF-QKV-TMA-4MMA
---
# SM120 FFN七段提交的有限接续实验

用户附件WAM_SM120_FFN_NEXT_CODEX_PROMPT_20261024.md明确授权本轮T0→T1→有条件T2→T3。仅在现有5090执行，原十小时历史收据保留，采用现有持续使用授权；不占用双A100、不改投机或质量门。假设与采用决定分列，当前实际保留A不变。

- [x] T0：冻结当前SM120 A源码/独立二进制，核对真实设备/资源/输入，逐段记录数值递推及已有B/C/D/E负结果矩阵。完成后保存并在对话打印完整结构/计时表，未重测标历史。
- [x] T1：独立诊断产物补FFN2 MMA、前序等待、转换/partial提交及发布成本；正常A/A和插桩扰动、shared容量核对。只在证据支持时选择相邻两段同owner一个候选，否则直接收口。
- [x] T2：至多一个设计和一次有定位修订。保持ACT、FFN1、M64/N128、各段K2048/K64与七次BF16舍入；先有限逐位/状态/七段递推，再三KV两进程正常完整block和FFN对照。失败/拒绝/修订均完整表。
- [x] T3：NET_GAIN / LOCAL_ONLY / NO_GO / INCONCLUSIVE分类、五问题、固定源/原始数据/采用决定、实际资源清理、HW和公开镜像。

## 资源及停止

T0只读/准备CPU≤2。T1新作业最多20分钟、T2最多40分钟，含编译/就绪/失败/清理；总新5090作业生命周期最多60分钟、每job最多4CPU、GPU单worker。这里为本轮有限工作上限，并非重置旧账或扩大其他资源。若T1已可判定则不启动T2。新增研究产物远端≤2GiB、本机≤1GiB，原A少量复制计入；清理预留各90秒。

预计当前会话内收口；结论充分即停止，不耗满额度。每个实际job保存目标、输入、软件/UUID、起止与硬截止、调用数、身份(PID+birth)、phase终态、存储、清理。CUDA active未知时明确未知，不把进程生命周期代替active。无可用无争用窗口只继续CPU部分。

## 契约与冲突核对

当前最强共同条件为eager、无CUDA Graph、warm CrossKV；附件中“复用当前warm Graph”与实际不符，本轮沿用当前三方无Graph条件。Graph能力若当前不支持，不通过新建Graph测量改变公平边界。当前实际SM数/grid从记录及现场核对，绝不沿用旧A100108。

不做smoke、不扫描模型SHA、不改阈值/容差/模型/权重/attention，不重跑旧M16/DSM/分工候选。有限源码/二进制身份按附件核对，用户禁止独立大文件SHA保留。冻结源码及字节核验可追溯；未知能力/计数器标UNKNOWN。

## 产物

SOURCE-MANIFEST.json、NUMERIC-CONTRACT.md、NEGATIVE-EVIDENCE.md、BOTTLENECK-AND-DECISION.md、REVIEW_RETURN.md、STAGE-TABLE.md、snapshots/、原始RESULT.json、CHANGE.patch及RESOURCE-AND-STOP-REPORT.json。独立源码/DSO放本namespace，不修改当前采用入口；当前A由既有Self-QKV-TMA采用记录和现场字节比对固定。
