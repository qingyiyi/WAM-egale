# 投机采样：单worker全层C/HC验证简报

2026-10-12T14:31:06+08:00（Asia/Shanghai）

- 状态：**FAIL**；通过关：尚无。
- GPU启动：False；target预算/计数：{"target_reserved": 0, "target_attempted": 0, "target_completed": 0, "stage": null, "stage_attempted": 0, "allocated_cap": 895, "target_hard_cap": 900, "extra_micro_calls": 0, "EGL": 0, "complete_episodes": 0, "complete_policy_chunks": 0, "environment_steps": 0, "counter_source": "no_GPU_attempt"}，硬限900；0episode/EGL/完整chunk/训练。
- 原batch14/原SDPA/公共模型不变；C/HC仍14行展开复用，非新prefix kernel，非TILE。
- 停止原因：RuntimeError('new-job CPU preflight failed: test_loader_cpu.py')

## 进展边界与下一步

这是同固定状态的完整verifier replay；不证明完整policy、成功率、环境步数或E2E。失败及所有原始样本保留，不补跑。
失败已停止，不补跑、不训练；先CPU检查本关失败/已有原始结果。任何修复后的新GPU尝试须另确认预算，不能沿用本次一次性授权。

[状态](status.json)；[CPU预检](../../../../review/OMISSIONS.md)；[合同](plan.json)。
