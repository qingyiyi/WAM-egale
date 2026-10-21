# 最近Execution checkpoint

## 2026-10-21T13:04:10.565362+08:00 M39控制实验完成，M40后台链自然完成

- 同源码独立/整层两进程三KV差约±1.4%；旧不同编译路径不能混比。静态local不代表已测动态spill。
- 专用FFN2一次候选通过9签名有限逐位，两进程完整层改善0.36%–0.65%，保留；仍慢15%–30%。未ready。
- M39本轮GPU81.559秒、CPU396.740秒，累计1852.227/36000秒，所有owned退出；[证据](../../experiments/single-layer-async-gemm-20261020/ffn-resource-control-20261021/RESULT.md)。
- M40 CURRENT18完整13成功5失败、审计/ON500updates/CAL11195全完成；原失败保留。13:02核验原后台链自然退出。DEV90尚未排队；[终态](../../experiments/robotwin-pilot-20261020/current18-followup-20261021/RESULT.md)。
- Maintain已合并用户“FFN先于ready、要求原因证据”及“RoboTwin继续后台”到既有constraint，不将机制猜测升格事实。
- [前执行记录](history/ffn-resource-control-20261021/EXECUTION.md)。
