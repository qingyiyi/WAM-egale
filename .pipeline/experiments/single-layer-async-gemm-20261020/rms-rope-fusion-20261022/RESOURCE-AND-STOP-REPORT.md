# 资源与停止

新用户授权RMS+RoPE融合，窗口2026-10-22T15:25:54.929267+08:00至2026-10-22T15:55:54.929267+08:00；实际2026-10-22T15:28:28.686414+08:00owned PID+出生身份全部退出，未发信号，GPU列表为空。旧截止没有修改。

GPU worker生命周期94.133秒；含上传、等待、清理整窗保守计费153.757秒，worker为子集不重复加。累计7480.439/36000秒，余28519.561秒。CPU编译308.493秒在GPU窗口前单列；CUDA active未知。

4项GPU步骤全终态exit0，无GPU失败或timeout。第一次只读收尾检查早于timing2结束，断言拒绝归档；之后等原作业结束，不重跑。见[CLOSURE-CHECK-NOTE](CLOSURE-CHECK-NOTE.md)。当前停止，不恢复过期RoboTwin任务。

本地编译及证据22.73MiB/512MiB，远端11.17MiB/128MiB。[终态](STATUS.json)、[清理](FINAL-CLEANUP.json)、[本机部署](DEPLOYMENT-LOCAL.json)、[远端部署](DEPLOYMENT-BAAI.json)。
