# 完整block GPU接续结果

更新时间：2026-10-19T09:03:43.594897+08:00。固定科学base ca5f37e，单A100-SXM4、CPU≤2，新预算起点08:48:46、绝对清理截止10:48:46；原CPU/M22登记未修改。

OBSERVED：六个保存的真实action block0输入（B2/B14×KV288/1152/2160），uncached与static_original_shape_context_projections两层，R0/R0_TWIN/R1/B0_TORCH/B0_ALIGNED以及B14 guarded强路径，共66次完整block执行。输出相对R0及旧保存GPU gold均有限且逐位相同，max_abs=0；全部有效KV/元数据、输入与当前CPU/CUDA RNG通过。cached严格原shape投影，不重新定义算法收益。66不是独立输入数，CPU另有六个合法写控制，不把它们称为本次GPU通过。

OBSERVED：attempt02只到首fixture A/A，三轮B/A为1.00744665、1.01964455、1.02512635；最后一轮2.512635%超预登记2.5%，按协议停止，不继续B0配对计时、不重跑挑PASS、不放宽门。实际B0性能仍UNKNOWN，不能签性能对齐。所有raw和FAILED保留。

OBSERVED：两次worker累计118次block、约43.484秒生命周期，均自然退出、owned出生身份/退出见各RECEIPT。CUDA active时间UNKNOWN。整个新阶段准备/等待持续计入同一120分钟/≤5000调用/≤1GiB预算，不能把worker累计当阶段总墙钟。

UNKNOWN：NVML在0MiB、无compute/pmon进程、约31.8W时仍报24–25%，根因未确认；原≤5%计时资源门保持false。后续CPU-only监控每60秒查询，只有资源真正变化、两次≤5%且0显存/无进程才最多启动一次原矩阵；原A/A门仍保留，截止不重置。原异常下诊断性A/A不提升为无争用正式性能。

RECOMMENDATION：保留B0数值阶段成果，优先澄清资源/计时稳定性，再补完整block性能；尚未开发persistent/tile scheduler或集成30层。原始capture已contiguous，原CUDA stride/alias/RNG、视频完整block、B0完整30层及完整policy仍不由本轮证明。原M22仅attention替换的模型/policy有限证据单列，不借用。

四项科学false、旧NO_GO、默认A_off、原tile与state/cycle/rules全部不改。原始索引：attempt-01/matrix/RESULT.json、attempt-02/matrix/RESULT.json、RAW-TIMING-SAMPLES.csv、WATCH-REGISTRATION/HEARTBEAT/RESULT、IMPLEMENTATION-MANIFEST与每个RECEIPT；新小源与父清单关联，未做模型/tensor独立hash。
