---
kind: decision
name: 真实B2与B14算子完整验证边界
level: reference
scope: operator-validation-20261018
status: active
updated: '2026-10-18'
supersedes: []
sources:
  - .pipeline/experiments/operator-kv-matrix-20261018/COMBINED-RESULT.md
  - .pipeline/experiments/operator-kv-matrix-20261018/SUMMARY-COMBINED.json
  - .pipeline/experiments/operator-kv-matrix-20261018/CPU-SAVED-OUTPUT-AUDIT.json
---

# 真实B2与B14算子的验证边界

本轮真实矩阵覆盖B2 video/action与B14 action、短中长历史、first/last及层0/8/29，共54点。standalone在全部单attention/block比较满足旧.02绝对加.02相对容差，但多数不逐位一致；六个真实B14输入上同时替换完整30层self/cross后，六例都非逐位，后段三例也未通过该容差，实际有效action CFG最大绝对误差.1875。旧六fixture有限验收仍保留，不能升级为完整Transformer严格等价证书。

既有shared-prefix与guarded路径只替换self，cross仍Torch；在本轮六个完整B14 query的原始与CFG输出、模型input_dict字节、全部有效KV及RNG范围逐位通过。该有限证据不覆盖所有输入、B2/video完整30层、scheduler/接受路径或机器人闭环。G1只测kernel，不能借用guarded完整query结果。

性能54点单进程3轮已采集，standalone attention核心总体接近Torch；完整block有14/54慢点且波动明显。启动时可见进程/显存为空但NVML利用率非零根因未知，全部延迟仍是探索性，不能宣称稳定对齐或完整任务加速。原双进程长重复31点PARTIAL未改。

精度与完整I/O已依用户授权在共享GPU单独验证，不混入性能计时；九组真实保存输入另做直接输入字节/布局/device/RNG检查，24份完整输出CPU独立核对一致。全部原始统计、失败与源码版本见证据目录，原tile代码和tile HW未改，四项质量/正式性能false不变。
