# Long500 四worker编排接续

仅改变非测速完整任务的资源编排：两张40GB A100各两个独立模型worker，独立环境/KV/RNG/scheduler，server/client各2CPU，总16CPU；不修改TF64、权重、normalizer、tau、tree、PF2、O3或target数学。

原v1源码67d1c0d与其所有终态/FAILED保持。受控handoff只在两worker各自的case最后一臂时暂停CPU runner，让client/model自然完整运行；原FINISH闭合后退出空闲server，再让原runner写自己的PARTIAL与清理。新plan明确只读复用三臂完整原case，旧partial不自动重试。原48h绝对起止、系列6001000 forward与10GiB不重置。

每worker=(task+init)%4，physicalGPU=worker%2，保持同case原GPU；每case三臂顺序与原manifest相同。旧/新并发负载的计时分层，不合成正式性能。更高吞吐是待观察结果，不能因显存采样能容纳两份就宣称加速。

v2修复optional replay constructor重复worker_index的日志错误。v1的capture UNKNOWN/gap不更正；新三桶fixture仍未获得replay资格，不增加模型forward、不截episode。AutoDL不smoke/独立测试/SHA；所有对照使用相同完整512-token BF16优化reset bank。四质量/TEST/非劣/正式性能权限恒false。
