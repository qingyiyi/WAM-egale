# 投机采样6局真实GPU不扰动验证：通过

2026-10-11（Asia/Shanghai）。仅action-only投机采样，非tile。用户批准6次新episode尝试上限、90分钟GPU阶段安全限时及任务专用监督器/CPU比较器；已按T_ON→A_off_OFF→A_off_ON串行关卡完成，未重试/补局/训练。

## 结果

| 条件 | init3 success／步数／chunks | init4 success／步数／chunks | 本轮关卡 |
|---|---|---|---|
| T_ON（对旧受控T_OFF） | 成功／293／19 | 成功／300／19 | 全输出/已测轨迹及账本PASS |
| A_off_OFF（新基线） | 成功／283／18 | 成功／291／19 | 完整性/初态PASS |
| A_off_ON（对本轮新OFF） | 成功／283／18 | 成功／291／19 | 全输出/已测轨迹及账本PASS |

- 前置132 CPU tests通过：43专用＋76既有成本包装＋13公共回归；旧T自比较76张/595行，内存故障注入拒绝第1步state hash和未执行动作通道29差异，未修改参考。
- T ON/OFF 76个完整action/video张量对、595行指纹和native；A_off ON/OFF 74张、576行。合计150个完整张量对、1171行已测轨迹比较；两侧shape/dtype/torch.equal/数值字节SHA、全部30动作通道及video一致，不用allclose/L2放宽。初始20数组/XML/init-state/RNG/双RGB及终止/success/horizon/steps/chunks一致。
- ON原profile与sidecar一致；T每policy action50＋terminal1/video21；A_off ordinary K2七逻辑/十四CFG行，actual forward/attention全block/有效QKV/50 committed updates、接受深度和fallback账本通过。不把fallback提交计成额外forward，不加bonus。OFF未profile，不伪造其counter。
- T双worker同19 chunks，未覆盖单worker尾部；A_off自然18/19 chunks，真实覆盖2个single-demand尾部请求（最后cache与policy），正常结束、不截断/假peer。真实server区间重叠不等于GPU kernel重叠。reset/cache内部phase仍未知。
- 第二次CPU读出复核PASS，CUDA未初始化；783 source/reference/input hashes、771既有immutable文件、9 target size/mtime未变；公共/原成本guard/权重/阈值/protected/旧raw/runtime及tile历史保留。各轮Knowledge、4类索引、compact、PROGRESS、RESEARCH与英文log已增量记录。
- GPU阶段24.13分钟，低于90分钟限时；6/6尝试、6个完整episode、0重试。12个owned server/client PID退出、5端口释放、GPU共享锁释放，显存0、无可见compute apps。后台tmux运行不依赖本机熄屏/SSH保持；不保证服务器重启。

## 结论边界

通过的是这两个受控init/冻结配置下profile＋成本包装＋cohort的组合不扰动，不是全hidden KV/server RNG证明、成功率统计非劣或自然整局加速。A_off这两个首局比T少10/9环境步、少1/0 chunk只是小样本描述，不与旧287/289步混算，不据此晋级或宣称真实环境训练优于离线。

## 下一步（提醒）

本关及CPU二次复核已结束。下一步先CPU汇总现有T/A_off成本账本的物理阶段/输入shape/有效KV/需求/重叠与匹配覆盖，再由用户单独确认最多8局同负载成本预算。profile-off无屏障的自然E2E和困难阶段冻结target训练继续后置；暂不teacher/训练/RL/KL训练/大小模型联训，不自动启动新的GPU任务。

## 原始证据

[本任务job](../../../投机采样/results/runtime/p0-gpu-nonperturb-six-20261011-152919)；[T关卡](../../../投机采样/results/runtime/p0-gpu-nonperturb-six-20261011-152919/gate-T_ON.json)；[A_off OFF完整性](../../../投机采样/results/runtime/p0-gpu-nonperturb-six-20261011-152919/gate-A_off_OFF.json)；[A_off ON/OFF](../../../投机采样/results/runtime/p0-gpu-nonperturb-six-20261011-152919/gate-A_off_ON.json)；[CPU二次读出](../../../投机采样/results/runtime/p0-gpu-nonperturb-six-20261011-152919/CPU-SECOND-READ.json)；[GPU收尾](../../../投机采样/results/runtime/p0-gpu-nonperturb-six-20261011-152919/OWNERSHIP-CLEANUP.json)。HW各轮见RESEARCH第24–30节；知识事件CPU/START/T_ON/A_off_OFF/A_off_ON/FINAL/POST_CPU均已记录。原第23节待确认方案和旧8局prepare guard保持历史/默认关闭状态，不把已获6局预算扩成8局。
