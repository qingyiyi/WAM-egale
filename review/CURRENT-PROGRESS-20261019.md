# 先读本轮CPU结果，固定当前审查commit

当前源码/HW由服务器镜像到main；一次网页审查请固定所读提交。数学参考base仍为ca5f37e0482c2e38e3738e3d8e71e3b185990a23，新CPU结果不追溯改写该提交。

1. [本轮A–E结果](../.pipeline/experiments/wam-block-baseline-20261019/RESULT.md)
2. [整体进度与时间安排](../.pipeline/cycles/wam-research-20261016/STATUS-20261019.md)
3. [B0源码与可执行CPU入口](../投机采样/experiments/block_sequential_baseline_v1/README.md)
4. [契约/DAG](../.pipeline/experiments/wam-block-baseline-20261019/BLOCK-CONTRACT.md)、[覆盖](../.pipeline/experiments/wam-block-baseline-20261019/BASELINE-COVERAGE.md)、[来源/完整hash](../.pipeline/experiments/wam-block-baseline-20261019/SOURCE-EVIDENCE-PACK.md)
5. [数值字段](../.pipeline/experiments/wam-block-baseline-20261019/ACCURACY-RESULT.json)、[性能限制](../.pipeline/experiments/wam-block-baseline-20261019/PERFORMANCE-LIMITATIONS.md)、[readiness](../.pipeline/experiments/wam-block-baseline-20261019/SCHEDULER-READINESS.md)

已建立显式顺序host混合B0；六真实fixture与合法写CPU通过、22契约/负例通过。GPU matrix/worker只是准备代码，未运行；54个性能单元无数值。不能把CPU比对写成GPU对齐，把完整block封装写成设备tile primitive，或把旧R1有限性能当新B0性能。

旧paused72/RESULTS.json仍为历史；四项质量/TEST/非劣/正式性能权限false，默认A_off、原失败/retention不变。原PT/完整KV/权重/native二进制不上传，不能仅凭GitHub声称重算原tensor。
