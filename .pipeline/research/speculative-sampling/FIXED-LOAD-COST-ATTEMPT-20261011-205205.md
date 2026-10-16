# 固定负载成本第三次GPU尝试：私有资源监控误停

2026-10-11T21:26:49+08:00（Asia/Shanghai）

## 结果与边界

- 新job：[p0-fixed-load-cost-20261011-205205](../../../投机采样/results/runtime/p0-fixed-load-cost-20261011-205205)。不是旧失败job续跑。
- 221项CPU回归/直接原constructor-reset-draft-prefix入口probe通过后，用户确认预留独占窗口并空闲后自动启动。
- 三次空闲观察（21:09:21/39/58）为0MiB/无compute PID；21:10:36监督器资源门通过，输入1304 SHA/9target stat、端口/约59.86GiB磁盘/51.72GiB RAM检查通过。未把未知约25%背景util当独占实证。
- 两torchrun launcher启动并开始实际CUDA权重加载；尚无target/draft ready。21:11:01私有guard发STOP，21:11:05失败与清理记录。停止判定26.05秒，清理/HW后28.40秒；未到90分钟限时。
- EGL=0、固定完整chunk=0、新完整episode=0。INITIAL/FIXTURES/G1/四格成本全部未执行，**没有性能结果，也没有证明GPU完整启动通过。**

## 已证实的误停原因

我新增的stdlib私有资源guard只白名单监督器登记的launcher PGID；Torch实际为rank新建session，两个rank的PGID不是launcher PGID。NVML中的3588892/3588895是本任务子worker，不是他人的进程：

| 注册launcher | 本任务rank | 归属证据 |
|---|---|---|
| 3588788 | 3588892 | worker0 model.log，父launcher向该rank发送SIGTERM |
| 3588789 | 3588895 | worker1 model.log，同上 |

安装环境的Torch SubprocessHandler._popen直接使用start_new_session=True；新CPU测试执行原constructor/_popen AST，真实/proc父子链、独立PGID重现旧guard误报。并未在此次误停前观测到上两次mode/reset字段错误；也不能因此声称它们GPU验证已通过。

## 清理与历史保留

OWNERSHIP-CLEANUP PASS，四个明确launcher/rank PID均已不存在，本任务五端口不监听。误停后一次观察GPU=0MiB/无compute PID；21:20:42后另一个hlj进程3623855使用GPU，不能把后续全卡有显存判为本任务泄漏，也不干预它。全局显存不保证一直为0。

原START/FAILED event、STOP_REQUESTED、guard错误标签与完整日志保留，不覆盖为成功；1347文件复核含原1304冻结源/输入和本次失败job文件，全SHA不变，11原实现/9target stat/保护文件与cycle缺失状态均不变。CPU修复仅新增独立目录。

## 下一步

按失败停：不恢复/修改旧job，不自动GPU重试或训练。CPU私有监控v2已验证；若用户另确认新job与0完整episode/2EGL/168固定chunk/90分钟预算，重新预留GPU窗口和冻结CPU gate后，真实启动→两EGL初态→fixture恢复→G1→四格GPU成本。

[CPU修复报告](FIXED-LOAD-OWNERSHIP-CPU-FIX-20261011.md)；[CPU验证JSON](../../../投机采样/results/runtime/p0-fixed-load-ownership-cpu-fix-20261011-211415/CPU-OWNERSHIP-REPAIR-VALIDATION.json)。T/A_off成本始终拟在GPU测；CPU预检和日志比较器不是CPU模型测速。
