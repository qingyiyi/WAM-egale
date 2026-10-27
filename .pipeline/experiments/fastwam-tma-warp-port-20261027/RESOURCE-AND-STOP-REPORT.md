# 资源与终态

闭合：2026-10-27T05:50:51.629902+08:00。本轮4个job全部自然退出，累计生命周期278.660秒（4.64分钟）/90分钟，编译、准备与清理包含；不是CUDA active时间。未重置前轮账本。最后现场compute contexts为空。

|Job|终态|生命周期秒|owned清理|
|---|---|---:|---|
|benchmark-attempt01|COMPLETE|32.651|PID154653已退出|
|build-attempt01|COMPLETE|68.659|PID154347已退出|
|policy-validation01|COMPLETE|105.911|PID154556已退出|
|validate-attempt01|COMPLETE|71.440|PID154435已退出|

本轮无FAILED/timeout。validate/policy科学状态EXACT_PASS；benchmark COMPLETE且计时输入精度通过，不能只凭exit0判精度。剩余账面5121.3秒，但本轮提前闭合不自动续跑。
本机证据7.46MiB，远端目录现场27MiB，低于2GiB上限。无新增权重、模型整文件hash、NCU、驱动变更或新租卡。
停止边界：迁移/数值/同输入21阶段测量已完成；后续split-K并行、DAG或闭环集成未自动启动。旧数据/失败/原四科学权限保留。
