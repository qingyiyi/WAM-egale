# WAM-egale：当前研究进度镜像

本仓库用于让网页版阅读服务器上的当前代码、HW和研究证据，统一维护 **main**。历史阶段保留在提交历史中，旧固定commit仍有效。进行一次审查时，请记录所读commit，避免审查过程中混用不同版本。

## 当前入口

- [整体进度与下一步](.pipeline/cycles/wam-research-20261016/STATUS-20261019.md)
- [给网页版的完整Prompt](review/PROJECT-REVIEW-PROMPT-20261018.md)
- [当前审查导航](review/CURRENT-PROGRESS-20261019.md)
- [新完整block顺序基线：CPU结果、GPU待验](.pipeline/experiments/wam-block-baseline-20261019/RESULT.md)
- [Long首轮500组三臂结果](.pipeline/experiments/long-morning-20261018/FINAL-SUMMARY.md)
- [新算子数值对齐](.pipeline/experiments/operator-torch-alignment-20261018/RESULT.md)与[性能报告](.pipeline/experiments/operator-torch-alignment-20261018/performance/RESULT.md)
- [HW短入口](.pipeline/PROGRESS.compact.md)

Long首轮1500局已完整；Goal/Spatial继续、Object按既定队列等槽。TF64结构/API继续作为研究载体，权重/训练配方未取得原质量证书。旧SDPA替换有有限GPU数值证据；新显式顺序B0已通过六真实fixture/合法写CPU与22契约测试，GPU精度/性能未运行。旧完整block性能慢点与计时波动仍保留。运行进度以报告标注时刻为准，仓库不是实时监控。

engineering_quality_screen=false；independent_test_complete=false；noninferiority_established=false；formal_performance_permission=false。main合并和算子数值通过不改变这些门。

## 历史与公开范围

[旧START_HERE](review/START_HERE.md)、[RESULTS.json](review/RESULTS.json)和paused72等保留历史身份，不能当成当前覆盖或资源状态。旧FAILED、负结果、阈值、retention及科学基线commit均未改写。

不上传模型权重、原始相机/完整KV/大tensor、编译产物或凭据。公开源码和报告不等于完整可运行环境或新增执行授权。第三方来源与许可见NOTICE、LICENSE及各实现的来源文件。
