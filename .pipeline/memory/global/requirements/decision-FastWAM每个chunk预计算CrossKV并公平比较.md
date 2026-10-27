---
kind: decision
level: constraint
scope: cycle:wam-research-20261016;FastWAM算子缓存候选
status: active
source:
  - 用户询问已有prompt缓存是否允许直接读取cross KV，听取实现与核验方案后回复“可以的，我同意”
supersedes: []
---
# FastWAM每个chunk预计算CrossKV并公平比较

已批准先在policy chunk内预计算条件投影及每层Cross K RMS/V，后续去噪调用读取缓存。首版分别保留B1/B7原计算形状；跨batch复用需先证明数值/索引相容，不直接借用B1缓存替换B7。机器人状态每chunk更新，完整条件缓存不得无条件跨chunk沿用。

先核验原Torch eager精度、全部有效KV/RNG、真实draft接受路径和完整动作，再比较性能。Torch参考与自有算子同等使用缓存；首次构建成本纳入完整chunk，warm层成本单列。保留官方compile强baseline，不把公共条件缓存计为自研调度创新。

授权是有限实现和真实输入/完整policy诊断；不自动改投机生产默认、阈值或重跑质量队列。原始结果、前轮FP32未采用结论和四科学权限保留。具体预算和终态见fastwam-crosskv-cache-20261027。
