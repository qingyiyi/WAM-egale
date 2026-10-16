# 投机采样：CFG2七候选零拷贝共享KV

2026-10-12T16:25:28+08:00

{
  "status": "FAIL",
  "error": "RuntimeError('fresh shared-prefix CPU preflight failed: test_entry_cpu.py')",
  "traceback": "Traceback (most recent call last):\n  File \"/rjs/cxz/论文探索/投机采样/experiments/shared_prefix_gpu_v1/supervisor.py\", line 305, in run\n    result=self.execute_phases(mode)\n           ^^^^^^^^^^^^^^^^^^^^^^^^^\n  File \"/rjs/cxz/论文探索/投机采样/experiments/shared_prefix_gpu_v1/supervisor.py\", line 328, in execute_phases\n    if mode in ('cpu','auto'):self.cpu()\n                              ^^^^^^^^^^\n  File \"/rjs/cxz/论文探索/投机采样/experiments/shared_prefix_gpu_v1/supervisor.py\", line 183, in cpu\n    if child.wait(timeout=900)!=0:raise RuntimeError('fresh shared-prefix CPU preflight failed: '+script)\n                                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\nRuntimeError: fresh shared-prefix CPU preflight failed: test_entry_cpu.py\n",
  "timed_out": false,
  "budget": {
    "target_reserved": 0,
    "target_attempted": 0,
    "target_completed": 0,
    "stage": null,
    "stage_attempted": 0,
    "allocated_cap": 585,
    "target_hard_cap": 715,
    "extra_micro_calls": 0,
    "EGL": 0,
    "complete_episodes": 0,
    "complete_policy_chunks": 0,
    "environment_steps": 0,
    "counter_source": "no_GPU_attempt"
  },
  "passed_gates": [],
  "next_step": "失败已停止，不补跑、不训练；先CPU检查本关失败/已有原始结果。任何修复后的新GPU尝试须另确认预算，不能沿用本次一次性授权。",
  "cleanup": {
    "status": "PASS",
    "remaining_owned_pids": [],
    "occupied_owned_ports": [],
    "other_processes_not_signalled": true,
    "sample": null
  },
  "counter_recovery": null
}

下一步：失败已停止，不补跑、不训练；先CPU检查本关失败/已有原始结果。任何修复后的新GPU尝试须另确认预算，不能沿用本次一次性授权。
