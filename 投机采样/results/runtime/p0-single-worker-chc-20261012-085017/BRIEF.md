# 投机采样：单worker全层C/HC验证简报

2026-10-12T08:57:05+08:00（Asia/Shanghai）

- 状态：**FAIL**；通过关：CPU。
- GPU启动：True；target预留/尝试/完成：52/52/52，硬限900；0episode/EGL/完整chunk/训练。
- 原batch14/原SDPA/公共模型不变；C/HC仍14行展开复用，非新prefix kernel，非TILE。
- 停止原因：RuntimeError("worker gate failed: LeaseError('requires unmodified eval original WanAttention interface')")
- GPU加载至清理：1.26分钟（45分钟安全限时）。

## 进展边界与下一步

这是同固定状态的完整verifier replay；不证明完整policy、成功率、环境步数或E2E。失败及所有原始样本保留，不补跑。
失败已停止，不补跑、不训练；先CPU检查本关失败/已有原始结果。任何修复后的新GPU尝试须另确认预算，不能沿用本次一次性授权。

[状态](status.json)；[CPU预检](preflight/CPU-GATE.json)；[合同](plan.json)。
