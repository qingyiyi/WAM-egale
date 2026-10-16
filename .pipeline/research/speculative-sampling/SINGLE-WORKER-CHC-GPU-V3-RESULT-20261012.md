# 投机采样：单worker C/HC v3 CPU预检失败，GPU未启动

2026-10-12T14:36:10+08:00（Asia/Shanghai）

## 结论

**本轮新job已执行CPU预检并按失败停规则终止；0新GPU/队列/真实target前向/episode/EGL/完整policy chunk/训练。不是别人kill、不是已测CPU模型的数值失败，是我接入新版本时漏同步两个负例测试契约。GPU正确性和性能仍NOT_RUN。**

用户本轮指令“GPU空了，你直接开始做吧”按此前1 worker、900 target前向硬限、45分钟含加载清理、0episode/EGL/full policy chunk/训练解释为一次新job授权；未改共享模式、未信号他人。本次CPU失败后授权不用于补跑。

## 本轮做了什么与已测结果

新v3入口将已验证写前日志接入真实Runtime；调用前BEFORE/返回后RETURN均fsync，预算意图与host-return分开，不冒充CUDA设备完成；GPU监督器只读恢复绑定PID/UID/start身份，不拿父旧0替代缺失日志，也不推断整段PASS。GPU deadline在切换PersistentLedger后保留，R/C/HC同计时收费，公共模型、batch14、attention不改。

|CPU子验证|结果|
|---|---|
|新job监督器/入口/恢复|49/49 PASS|
|写前日志及实际worker终止/数值|44/44 PASS，包括9个实际serve终止场景和全30层29变化query BF16 R/C/HC持久化probe|
|真实生产loader|35/37 PASS；2个CPUOnlyEntry负例的断言失败；7个生产加载/数值正例、20个坏接口拒绝、6个前置准入等均通过|
|原81项接口全回归|本job NOT_RUN；不拿旧结果当新通过|
|单独同worker FP32 CLI probe/新CPU gate|本job NOT_RUN|
|GPU恢复/捕获、全层C/HC验证、OFF/ON|全部NOT_RUN|

实际运行130项，128通过、2失败，0错误；所有CPU结果均CUDA_initialized=false、禁止CUDA尝试为空。CPU controller开始到FAILED事件245秒（约4.08分钟，含子进程加载和失败清理，不含代码编写）。GPU-ATTEMPT/入口/worker readiness/CPU-GATE不存在，所有本任务CPU进程清理PASS；当前不再等待或自动启动。

## 两个失败的准确原因

1. **旧job的CLI拒绝负例没有清理v3 flags。**逐字复制的旧测试只移除`P0_CHC_V2_`，但新监督器注入`P0_CHC_V3_`；于是入口仍正确拒绝旧job，但返回`scope mismatch: execution_revision`，不是测试预期的`default-disabled`。这是负例环境未隔离，不能放宽GPU入口来“让测试通过”。
2. **私有CPU子夹具的新报错类别未同步。**旧job路径被正确拒绝，实际报错`new private CPU fixture required`，旧断言只匹配`scope mismatch|new private job`。属于任务路径报错契约不一致，不能把测试失败掩盖成通过。

这是我新包装的工程遗漏。不会把这次FAIL改成PASS，也不会绕过CPU gate或继续GPU；冻结v3源码、所有失败日志保持不变。此处是只读定位，没有修复或重测。

## 不变与边界

657份冻结文件（含旧源码/证据和147份旧HW raw）、21份已执行私有源码及tested-source快照保持SHA；8份接口/数学/旧测试逐字复制，旧GPU Runtime的14个数学/API方法AST不变，已验证CPU日志主体只改显式启用guard导入。target九权重stat、A_off SHA、fixture stat及公共模型/server保持不变；state/rules/cycle未改，TILE旧记录保留不混写成果。自审非独立审计。

## 下一步

本job已停止，不补跑/复用/排队。下一步建议单独授权仅CPU同步这两个负例的版本flag清理和私有路径报错契约，补真实CLI/旧job拒绝回归；不放宽门、不改公共模型/batch/attention。CPU修复通过后，再单独确认新GPU job及独占窗口/原900次45分钟预算。

目标仍是证明长历史单worker全层C/HC复用不扰动且降低包含构建/staging/检查/释放/持久化的完整verifier净成本；此轮未取得真实GPU速度结果，更不证明policy成功率、环境步数或E2E。

[本轮失败](../../../投机采样/results/runtime/p0-single-worker-chc-v3-20261012-142632/FAILED.json)；[49项](../../../投机采样/results/runtime/p0-single-worker-chc-v3-20261012-142632/preflight/job-tests.json)；[44项](../../../投机采样/results/runtime/p0-single-worker-chc-v3-20261012-142632/preflight/counter-tests.json)；[两个失败](../../../投机采样/results/runtime/p0-single-worker-chc-v3-20261012-142632/preflight/loader-tests.json)；[清理](../../../投机采样/results/runtime/p0-single-worker-chc-v3-20261012-142632/OWNERSHIP-CLEANUP.json)；[只读审计](../../../review/OMISSIONS.md)；[已执行源码](../../../review/OMISSIONS.md)。
