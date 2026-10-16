# 投机采样：单worker全层C/HC GPU v4（默认关闭）

这是旧GPU v2与已验证CPU持久化计数的独立新修订；旧源码、CPU证据、GPU失败job不改。须新v4 flags、新job计划SHA、新job CPU gate、监督器单次attempt及本PID入口claim，才能导入真实GPU模型。当前用户允许一次1 worker、<=900 target前向、45分钟含加载清理、0完整episode/EGL/full policy chunk/训练。

## 数学与成本

R为原始路径；C/HC仍14行展开历史KV复用，不更改公共模型/batch/attention，非共享prefix kernel、非mega、非TILE。原30层29变化query的capture/sequence/invalidation/isolation/probe/diagnostic数学方法保持不变。

每次target forward调用前BEFORE、返回后RETURN，均fsync且位于原完整sequence计时内，所有R/C/HC同收费。BEFORE是保守预算意图，RETURN是host forward返回，不冒充CUDA完成。正常/TERM退出写CLOSE；KILL可只读恢复最多一个未确认意图。缺失/损坏日志不套父旧0，不退款、续跑、推断整段PASS。恢复还核对writer PID/UID/start_ticks与固定worker身份。

## 安全和执行

新job先51监督器/入口/恢复单元、44计数及真实worker终止、39真实loader、81原接口回归与同worker FP32 probe（BF16在44项中）。本修订仅修负例的新版本flag隔离及精确私有路径报错断言，补四种真实CLI在Torch/公共server导入前拒绝证据；不放宽任何GPU数值/资源/性能标准。CPU小随机checkpoint不是trained target GPU证明。CPU测试fault只允许绑定本job的CPU子夹具，禁止混入GPU计划。

GPU仅1常驻模型worker，从spawn到cleanup总限45分钟，原895分关预算/900硬限不变。先恢复/捕获→数值/十二类失效/13行隔离/显存→warmup及六平衡块OFF→独立ON→清理后CPU汇总。OFF冷构建/staging/键检查/释放/写前日志成本全部收费；ON不作正式测速。失败停止、不补跑、不训练。

共享Default模式不改；启动前若忙，每180秒只读等待，空闲后15稳定样本才启动，等待不计GPU限时。启动后外部GPU进程、显存/磁盘不足或超时触发只清理本任务。轮询不等于硬件预约；不信号外部用户、不sudo、不占满显存抢卡。

launch --mode auto使用独立session、忽略SIGHUP、stdin DEVNULL、stdout/stderr持久化文件；后台CPU全通过才自动交接GPU（GPU入口仍重新校验新CPU gate），CPU失败不交接，GPU只有一次尝试。与本机熄屏无关。用户已授权持续修复和并行只读讲解，本轮累计仍≤900 GPU前向/45分钟含加载清理，不将新job当成重置累计预算。每关hw_event.mjs增量写中文HW/PROGRESS、English log.yaml+配置log.md，保护state/rules/cycle并保留TILE历史。最终仍不是完整policy、环境步数/成功率/E2E证明，有稳定净收益才另规划闭环预算。
