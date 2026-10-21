# 当前执行证据（2026-10-21T15:47:48.199638+08:00）

用户要求投影等待warp参与MMA，本轮已试两个Cross K/V 8warp方案：大tile256×128完整层慢3.16%–4.85%；固定128×128只改warp映射仍慢1.05%–1.33%。两个独立进程/三KV方向一致；各9真实签名/23中间量、有效KV、输入/RNG有限逐位通过max_abs0。两机活动源码和远端DSO已恢复原4warp df2a307241928567e2b0d80a5e6c601b8e1aedaf6e4713b693d88ce247ead11a，GPU0释放。

完整层计时不能改称投影阶段计时；本轮没有新阶段/NCU。大tile增加shared，固定tile不增，但具体MMA/shared/local因果仍UNKNOWN。小action投影未试，不外推。后续先同前序阶段归因，再真实DAG补位。等待warp数量不直接等于可回收时间；没有异步warp专职/ready调度新成果。

- [8warp完整结果](../.pipeline/experiments/single-layer-async-gemm-20261020/projection-eight-warps-20261021/RESULT.md)、[逐进程计时表](../.pipeline/experiments/single-layer-async-gemm-20261020/projection-eight-warps-20261021/TIMING-SUMMARY.csv)、[回退](../.pipeline/experiments/single-layer-async-gemm-20261020/projection-eight-warps-20261021/SELECTION.json)
- [资源及终态](../.pipeline/experiments/single-layer-async-gemm-20261020/projection-eight-warps-20261021/RESOURCE-AND-STOP-REPORT.json)：本轮GPU79.587秒，M39累计2073.413/36000秒；0异常/超时，2性能负结果保留。
- [上一轮NCU/阶段诊断](../.pipeline/experiments/single-layer-async-gemm-20261020/remaining-cost-20261021/RESULT.md)：只对应旧4warp，不能移植到8warp候选。

RoboTwin固定DEV90当前完整33/90，状态RUNNING_COMPLETE_DEV_EPISODE，当前open_microwave-valid15-P；[STATUS](../.pipeline/experiments/robotwin-pilot-20261020/dev90-20261021-r2/STATUS.json)。microwave valid14三臂均失败，basket valid15只有PF2失败；初态/首两S等按原协议核查，不自动重试。GPU1后台继续，24h累计和10/22 01:26截止不变。

四科学权限false，LIBERO/当前DEV非独立TEST；baseline/原tile/投机生产/保护文件不动。公开小证据和被测源码；原始输入/权重/DSO留服务器。旧paused72及review/RESULTS.json不修改。

补充只读核查：[Torch真实CTA/warp与persistent区别](../.pipeline/experiments/single-layer-async-gemm-20261020/projection-eight-warps-20261021/TORCH-WARP-ANALYSIS.md)。B14小投影96CTA/4warp、Cross K/V各672/8、FFN1为112/8、FFN2为96/4；672非同时驻留，当前尚无同计算体CTA数单因素对照，不能把原版优势归于CTA总数。MegaMoE固定角色与动态任务区分见该文；未新增GPU或改算子。
