---
kind: decision
name: 投机重组与多候选协同调度
level: constraint
scope: project
status: active
updated: '2026-10-16T05:21:36+08:00'
supersedes:
  - 'decision-双路线与算子保护.md 中两路线松散并列及等待网页规划的安排'
sources:
  - '2026-10-16用户完整执行Prompt与夜间继续指令'
  - '.pipeline/research/speculative-sampling/WAM-COORDINATION-20261016.md'
---
# 当前研究主线

核验“投机重组小action工作量→多候选verifier可调度→新系统路径超越已有批处理→质量约束下实际成本降低”。这是待证伪因果链，不能预写为论文结论。TF64结构可暂冻，最终draft依质量/成本选择，保留等容量MLP。

先真实shape与profile，再选至多两条路线中的一条最小原型；比较当前shared-prefix/O3及合理Graph基线，不能将普通batching、已有prefix/suffix softmax合并当新贡献。保留每轮合法提交数g和全部draft/pack/verify/accept/fallback成本，不以候选数代替g。

原tile代码/HW、撤回融合、原FAILED、冻结协议、保护文件及全局A_off仍保留。新算术需逐级数值/状态及闭环门，TF质量false不开放正式45chunk。HW刚完成v15升级，本轮不再升级/回退；原v12安装仍存。

夜间先有限机制核验与质量诊断。AutoDL最新硬截止10/16 09:00，接续从CPUbank前起仍累计4小时含等待、预留300秒清理，无续租。旧100FAILED只可归档诊断，新命名空间资源接续不能伪装旧PASS或复用无targetRNG的旧T。

10/22 go/no-go、10/23冻结、10/28主表冻结、11/01 18:00完整中文稿交老师。写作在会话/用户飞书，不创建本地论文正文。

阶段结论（10/16晨）：H1/H2已有有限机制证据，路线B在共同缓存基线下有局部增量、完整policy/cache重放通过；当前TF扩大研究T48/50、TF44/50且max长度比1.4，H4未成立。主攻路线B，暂不展开路线A或大融合；默认A_off不变、TF正式性能仍阻塞。一次新数据/训练继续HOLD；先严格病例诊断及系统缓存生命周期/独立复测。证据统一见[夜间结果](../../../research/speculative-sampling/OVERNIGHT-RESULT-20261016.md)，不在此复制长报告。
