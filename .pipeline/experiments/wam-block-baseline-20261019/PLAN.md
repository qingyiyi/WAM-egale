---
kind: experiment
name: 完整block显式顺序基线
cycles: [wam-research-20261016]
status: active
review_commit: ca5f37e0482c2e38e3738e3d8e71e3b185990a23
---
# 先建立完整 block 基线

用户提供网页方案并要求本轮夜间先做、早上9点汇报，随后要求同步HW算子进度。本轮按照北京时间2026-10-19 09:00交付，执行入口为CPU_AUDIT_AND_BASELINE_PREPARE；附件明确未新增GPU授权。

- [x] 固定ca5f37e读取与源码/旧证据审计；分别列替换范围和验证范围。
- [x] 核实真实block、参数/运算/舍入/cache契约及现存fixture，事前最多6个核心case。
- [x] 在独立 `投机采样/experiments/block_sequential_baseline_v1` 实现R0/R1/B0与状态恢复、验证和计时代码。
- [x] 运行受限CPU契约/负例及可行的真实fixture CPU检查；区分CPU证据与GPU待验证，不冒充通过。
- [ ] 输出coverage/contract/accuracy-pending/performance-pending/readiness/资源/来源包，更新HW与GitHub main。
- [x] 9点前刷新远端既定队列：10/19 02:03 Goal336/500共同组、1010/1500局；Spatial340/500、1023/1500；Object=WAITING_NATURALLY_FREED_REMOTE_SLOT_CURRENT14；本轮CPU报告提前完成，发布同步收尾中。

CPU≤2核，CPU作业累计墙钟≤8小时，新增文件≤1GiB；08:30停止新CPU作业、08:50清理、09:00汇报。GPU=0、CUDA context=0、模型GPU加载=0；无模拟器/训练/新episode/迁移/生产patch/persistent scheduler。原tile与state/cycle/rules、旧FAILED/阈值/retention、四false均保留。仅按本轮来源要求计算小型源码/补丁指纹，不读取大权重/tensor做hash。

R0真实原block；R1已对齐attention替换的混合block；B0显式分阶段混合顺序执行，不是完整自有CUDA。设备D0/P_seq/P_sched仅列缺口；GPU120min是待申请建议，不借旧M22余额。远端M17原队列/巡检继续，不注入本轮代码。
