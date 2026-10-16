# 共享 KV：真实短档与中等档补测

用户已授权补齐短 KV 和中等长度。仅新增私有实验包装，原 Torch 参考、共享 CUDA v2、adapter 和旧失败均冻结保留；公共模型/server/config、受保护 state/rules/cycle 不改。本轮不实现文本 prompt 共享。

## 条件与顺序

- 短档：现有真实 T-init3-chunk1 快照，frame4。请求前历史144，视频缓存更新后 verifier 历史272，当前16，完整 KV288。历史固定条件原 A_off 记录为27个 verifier 查询、21 video＋2 ordinary action＋27候选＝50次非terminal前向；重捕获必须匹配27/50。
- 中档：恢复同一真实短档起点，按原路径回放已保存 frame4、8、12、16、20、24 的视觉 terminal、动作及观察；每段3次 cache 前向，共18次。动作经过原 postprocess→preprocess。得到 chunk7/frame28 的请求前真实历史1008，verifier 历史1136，完整 KV1152。没有裁剪长缓存、合成随机历史或新增环境运行。
- 两档均原 Torch capture；保留实际全部变化 verifier 输入及历史于 RAM，停止在 action terminal 进入 Transformer 之前。中档查询数依原 selector/threshold 自然产生，上限48；视频21＋普通 action最多2＋候选最多48，capture最多71。查询数不同，不声称跨档总延迟差全由 KV 长度造成。
- 每档 R 原 Torch / S 共享各全部查询预热，六组 R→S、S→R 交替，各臂覆盖完整30层和该档全部自然查询。直接继承冻结 Runtime.sequence：staging、lease/setup/keys、fsync journal、release 均在计时内；历史恢复、CPU输出检查/保存在计时外。无采样剔除、无性能重跑。
- 每次实际输出与同档原 Torch gold 逐位一致；all30 history/metadata/RNG、lease释放及显存检查保留。不同长度的原 Torch gold 分别保存；长档旧 gold 不改。原30层/29输出、12失效及隔离证据冻结复用。
- 两臂 MAD/median≤7%；positive：S/R中位≤0.95、六组均<1；negative：≥1.05、六组均>1；否则 flat/mixed。数值不一致立即失败，不放宽阈值。非稳定 positive 时，在该档 OFF 全部完成后，仅追加首个 query 的 R/S 各1次 profile用于诊断，不能替代正式样本。
- 每档 resource probe1次、finish R/S各1次。保存正式12份实际输出，独立 CPU 字节复查。已通过10项CPU检查的自有 root 僵尸态分类修复接入新监督器，活进程/PID复用/UID/PGID/祖先链不放宽；退出分类不代表工作成功，仍须完整结果、CLOSE journal、NVML及端口清理。

## 可审查的硬预算

短档最多50＋1＋2×27＋12×27＋2诊断＋2收尾＝433次；中档18历史回放＋71＋1＋2×48＋12×48＋2诊断＋2收尾＝766次；本轮总上限1199次 target写前意图，0直接attention。

1次GPU启动、1个worker、无重试，含加载/回放/capture/IO/同步/清理最多900秒。已用968次、1636.3243473544717秒，最坏累计2167次、2536.3243473544717秒（42.27分钟）。累计GPU仍保持2700秒（45分钟）。

**旧累计target上限1000只剩32次，GPU执行须另获用户明确确认，将累计target上限调至2200，并写入绑定本 plan 哈希的预算回执。** `--prepare`只能CPU准备，`--job`没有回执必须拒绝。本轮前向预算是保守上限，中档实际自然查询较少时不填满、不返还旧失败消耗。旧1000/900计划和回执不覆盖。

0 episode、0 EGL、0完整policy chunk、0环境动作、0训练。单worker固定 verifier 性能测试不属于正式并行EGL数据采集。不能由这两档实测声称所有长度都快或任务成功率/E2E已改善。

执行和实现审查由同一主代理完成，项目 execution.mode=self；独立 CPU 审计脚本验证 bytes/计数/冻结 SHA，不宣称独立代理复核。
