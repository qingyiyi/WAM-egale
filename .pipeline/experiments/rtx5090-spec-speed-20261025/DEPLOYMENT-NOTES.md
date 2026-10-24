# 下一次5090临时复测的已知事项

本轮实际故障与修复，不是新增实验授权。结果与清理以最终报告/收据为准。

1. 权重、BF16 full512 reset bank、draft/normalizer与冻结输入manifest一起按套迁入。先按清单检查文件及符号链接，使用rsync-aL或tar dereference复制真实内容。Object原bank为A100绝对链接，普通tar保留链接曾导致0teacher模型准备失败；原失败保留，实际bank复制后新attempt接续。
2. 原源码中资源根/model_source要在克隆函数globals或建立O3之前绑定到当前独立根。Long曾在首个动作前因旧绝对路径失败21teacher。数学/阈值/生产API不改。
3. 观察transformer.forward时保留原公开签名；原lease controller依赖它识别candidate_cache_rows。丢失签名曾触发正确的lease撤销，23尝试/22返回、0动作；不得删旧断言绕过。
4. 共享Q16/Q32扩展从本轮actual source/DSO复用时仍需绑定同Torch2.9/cu130/SM120环境。首16次真实Q32 inline参考审计只证明对应调用；正式计时应先完成数值核验，再单独登记不带该审计的测量。当前数据没有事后扣除它。
5. 原torch server与Python3.10模拟器的ABI/包版本要分别固定。CuRobo0.7.8五个native模块需要SM120重编；本轮只改CPU构建并行度/头文件与资源路径，数学源码与实际A100安装的117文件逐字节比较。完整实际runtime源与构建日志已回传，不能用版本字符串代替源码身份。
6. 原生rsync append时，文件系统曾在EOF之外预分配，导致物理占用短时超过本轮34GiB cap；修复为最终大小preallocate+物理du guard。4GiB prefix先posix_fallocate并核对实际写入offset，最终仍须native append-verify和fsync后才加载；所有等待计入原生命周期。
7. 全部三臂使用相同full512 BF16 bank；不用T5编码器时不加载其约10.6GiB权重。不能将共同reset优化算作投机贡献，也不能沿用旧reset/环境时间占比。
8. RTX5090上的GPU计算单worker；本轮有在独立CPU核预传下一套资产的时段。正式性能需要隔离IO、分离数值审计与计时并做独立重复；新完整机器人任务不截断、不自动重跑，不以单chunk或calls替代任务E2E。
9. 下一次可复用A100上既有导出包和本机回传的小源码/DSO，重新迁入5090。现有operator环境保留；本轮新venv/simulator/cache/权重均随临时根删除。不要恢复旧job的起点、deadline或quality权限。

CPU预检只核清单、导入/ABI与实际源绑定，不运行用户禁止的smoke或虚拟短episode。任何数值、质量或正式性能放行都需要它本身的证据；四项false不因迁移成功而变更。
