---
kind: decision
name: Object只补采输出冲突的14局并接续原任务
level: constraint
scope: cycle:wam-research-20261016;experiment:libero-four-suite-20261017;object_capture_recovery
status: active
updated: '2026-10-19T09:00:07.445964+08:00'
supersedes:
  - 本次Object row16至29补采待授权状态
sources:
  - 用户明确回复：允许补采14局并接续原任务
  - 已展示隔离目录补丁、固定14局清单及最多56000次target forward的请求
---

# Object固定14局基础设施补采已批准

用户批准在独立namespace补采因compact文件编号冲突导致监督无效的Object row16–29。保留旧16有效数据与全部旧14终态/无效采集记录，不重写FAILED或伪装原数据已通过。固定task/init、CONT3000权重与原FIT/CAL角色；最多14完整episode、每局最多4000 target尝试、新增最多56000次调用，无自动重试。原M17预算起点、绝对截止与总存储上限不重置；实际过去消耗与新增均记账。

补采使用新job专用的compact子目录，保留原监督/数值/容量/数据准入断言，不放宽训练资格。随后接续早已授权且未启动的ON3000与Object500组三臂，不新增测试臂/架构搜索/独立TEST或正式性能放行。可以使用空闲BAAI双卡独立worker，所有远端写入仍限~/cxz。小源修复与运行结果分列，只有实际验证通过才记录通过。
