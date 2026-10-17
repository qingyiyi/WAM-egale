---
kind: decision
name: PF2源码执行与AutoDL十二小时窗口
level: constraint
scope: experiment:tf-pf2-autodl-20261017
status: active
updated: '2026-10-17T11:11:41.228785+08:00'
supersedes:
  - 'PF2只读方案的execution_authorization=NONE（仅本轮后续实施阶段）'
  - '新双卡原申请累计5GPU生命周期小时资源上限'
sources:
  - '2026-10-17用户回复同意具体PF2执行/迁移/准备预算申请'
  - '随后用户明确你可以在12小时内无限使用autodl'
---

# 批准PF2源码执行，AutoDL可用十二小时

用户已批准独立源码实施、必要环境/资产迁移及冻结TF_ON连续/PF2六pair完整DEV；本轮资源窗口从首次记录北京时间2026-10-17 11:11:41起12h，清理硬截止23:11:41，提前15min停止新工作。两卡在此窗口内可用，不再受5累计GPUh帽限制；原5h申请不作为当前余额。最多两独立worker，同pair同GPU、三波全局止损。不做draft DDP，不新增租机。

原12完整episode/49000forward/3GiB新证据及每局1200s/4000尝试、固定六pair、原权重/norm/tau/selector/树/bonus/数值边界保持。资源许可扩大不等于新增训练、FIT、PF3/4、替换case、独立TEST、120或正式性能授权。首次准备≤2h计入12h；失败/等待/清理均记账，不重置旧attempt/deadline。

AutoDL无smoke、独立测试或SHA（包括loader/fingerprint隐式调用）；直接在独立namespace使用源码运行完整任务，必要审计随任务完成，未测门UNKNOWN。所有原FAILED、22清理缺证、NO-GO和四项false保留；保留所有失败，不因反向成功改旧门。不保存凭据；只清理自有且PID/出生时间吻合的进程，脚本退出不等于实例关机。
