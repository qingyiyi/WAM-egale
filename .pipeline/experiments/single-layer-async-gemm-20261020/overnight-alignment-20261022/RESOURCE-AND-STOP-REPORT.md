# 资源与停止记录

更新时间：2026-10-22T02:46:38.435053+08:00；状态：NCU_WINDOW_PENDING。

远端GPU阶段生命周期 1427.784780 秒；CPU构建/准备 2460.964687 秒（不是全部CPU分析时间）。远端owned在checkpoint均退出。

本轮保守计费 2312.197278 秒，包括本机自然空闲排队；原M39累计 4850.876801/36000秒，剩余 27687.802722秒，09:00硬截止更早优先。排队计入预算但不表示占用了GPU。CUDA active时间未测。

|本机作业|状态|含等待生命周期秒|GPU worker秒|guard仍存活|
|---|---|---:|---:|---|
|local-ncu-08|FAILED_OR_PARTIAL|0.064|0.000|False|
|local-ncu-final|FAILED_OR_PARTIAL|286.366|15.047|False|
|local-ncu-final-r2|WAITING_NATURAL_IDLE|597.983|0.000|True|

12个实现候选及08b正确性修复已终态；仅11及其前序有效改动进入最终版本。较慢候选、数值失败、测量包装器错误、NCU外部占用中断均留存；失败和等待计预算。未启动ready-DAG、30层融合、新机器人任务或扩大质量测试。

资源失败清单见JSON；初始Self-Q/point计时包装错误、两个native元数据wrapper错误、最终stage格式抽取CPU失败分别保留独立终态。脚本returncode0不覆盖guard对外部干扰的FAILED_OR_PARTIAL。

当前本地证据 76.78MiB，远端checkpoint 395.73MiB；总新增上限3GiB。模型/fixture没有逐候选复制。

四项科学权限false；关闭本轮作业不等于M39性能完全对齐，不等于RoboTwin扩大获准。
