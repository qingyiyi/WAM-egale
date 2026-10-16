# 投机采样同负载 T/A_off 成本包装：实现及CPU关卡完成

日期：2026-10-11T14:53:07+08:00（Asia/Shanghai）。范围：**action-only speculative sampling，非tile**。当前用户只授权实现和CPU验证，并明确禁止新GPU任务或训练；本轮严格按此范围执行。

## 1. 这一步回答什么

此前夜间 A_off 四对评测少27环境步、仍72 chunks、E2E反而+9.67%，且worker完成时间导致实际需求不同。因此先把请求调度、实际target工作、诊断成本与自然E2E分开，不能仅以更少L2/步数/更高接受率判断收益。上一关T/S0同init结果保持：init3 293/293，init4 300/600；init4 +300步中+280集中第二罐建立接触窗口，是阶段定位，不是唯一L2/z机制证明。

本轮**实现了真正的任务包装并通过CPU合同**，不再只是旧成本草案中的队列模型。仍未获得新GPU成本或真实LIBERO轨迹。

## 2. 实现内容

源码位于 `/rjs/cxz/论文探索/投机采样/experiments/cost_matched`：

| 组件 | 当前实现 | 边界 |
|---|---|---|
| 请求cohort | 两独立worker按reset/policy/cache及顺序配对，loopback协调器；异常、缺peer、错配时共同停止 | 到达队列同步，不保证GPU kernel重叠 |
| 完整尾部 | peer正常完成后剩余worker继续完整episode/horizon，标记single-demand | 不补假请求、不截断、不重试加局 |
| client账本 | 原请求浅复制，仅增加sidecar context；记录屏障等待、transport起止、请求id/任务/init/worker/provenance | 原累计policy计时含屏障，不冒充模型compute或clean E2E |
| server账本 | 进程内包装原infer/profile finish/transformer forward；保持原调用和返回值 | 公共server/model/client源码及默认行为不改 |
| 真实网络shape | 每forward分支、cache-update、CFG batch、输入shape；各block实际self-attention Q/K/V | 在有效history选择之后读shape，不以cache pool容量冒充KV长度，不读tensor值或新增CUDA同步 |
| 成本对账 | 原phase.calls/counters与实际forwards/rows分别保留；顶层与嵌套phase不重复加 | 无profile/无实际sidecar保持null；reset/cache内部phase未知仍未知 |
| 离线汇总 | request-id/时间边界/实际call/计数对账；显式物理阶段+frame/KV/需求/observed overlap分层 | 无标签/缺KV/无对侧→unmatched；保留尾部与完整reset/cache，拒绝覆写结果 |
| 默认运行闸门 | prepare-only，不导入真实WAM server或sim；默认GPU批准=false | 全8局执行还要求单独授权、CPU PASS及GPU完整轨迹不扰动PASS；本轮GPU关卡NOT_RUN |

预备合同仅保留候选 **T→A_off→A_off→T，每臂init3/4双EGL首局，共8完整episode**。普通K2、冻结A_off、冻结target/global threshold/原cache/torch attention/seed/完整horizon不变；无bonus/teacher/full trace/训练/RL/联训。prepared计划是CPU临时产物，不是GPU执行许可；继承的原20局训练入口没有调用。

## 3. CPU实证

最终统一命令使用现有 `libero-va` 环境，`CUDA_VISIBLE_DEVICES=''`、BLAS/OMP/MKL各1线程：

- **89 passed in 7.53s**：新增包装76 tests + 既有 `test_p0_minimal_frozen.py` 13 CPU回归。
- 两个真实CPU进程通过loopback broker配对；线程样例1/1、1/4、4/1、19/38、19/19、51/19保留所有reset/policy/cache和长尾。timeout/重复/错序/worker失败均停止，不隐式降级缺失peer。
- 正式公开client `_run_episode` 在FakeEnv/FakeModel中包装OFF/ON均15步、success与原请求相同；这不是实际LIBERO运行，也不替代真实轨迹开关验证。
- 从**当前公开源码**AST提取实际 `WanAttention.forward/custom_sdpa`，仅在CPU小张量执行；包装前后输出/RNG/cache mask一致。14候选行例子里cache pool容量12、valid history3、query2，记录实际KV5，而非12。不是完整WAM权重加载。
- 同样提取实际 `VA_Server.infer/_phase_profile_begin/_phase_profile_finish`，用CPU fixture验证原profile写出与sidecar完全相同、正常/异常finally行为、请求返回和RNG一致。原CUDA同步函数以CPU stub替代，不声称验证真实CUDA。
- 旧target-only真实profile **17行**重新只读解析：每行普通action50、action terminal1、video21；counters为空但不是零target工作。旧数据缺实际forward sidecar，actual calls/rows为null，不伪造GPU记录。
- ordinary K2账本合同验证CFG14、七逻辑候选、terminal及50 committed scheduler updates；拒绝bonus/错计数/缺counter/缺KV。**接受深度0/1时，fallback可直接用本次verifier已有target direction提交一步，不代表再跑一次transformer**；“committed更新”与“forward次数”分开。
- 离线汇总CPU合成fixture，无阶段标签时4条policy全unmatched，明确标签后2个分层/4条匹配；合成1秒数值仅用于合同，不是模型计时。保存样例均显式标注synthetic CPU/非GPU/非真实LIBERO。
- 独立复核732个既有公开源码/参考runtime/raw/protected等冻结文件SHA256未变；9个target权重文件size/mtime未变，A_off内容hash纳入冻结集；248项未来预备source/input manifest一致。9个新增实现/测试/说明文件单独hash。
- `torch.cuda.is_initialized()==False`；未导入真实WAM server/simulator，无新GPU任务、模型服务、真实episode、teacher或训练。测试的临时CPU loopback brokers已由测试关闭。不安装新系统依赖，不启动/停止无关tile任务。

首轮发现并修复pytest保留参数名冲突；一次误用无pytest的旧`libero`环境后改用已有`libero-va`，未安装依赖。最终统一回归日志才是最终验收计数，不以此前单套54/75/88项中间结果替代。

## 4. 仍未证明

1. 包装/profile在**真实GPU完整WAM输出/真实LIBERO轨迹**下不扰动：NOT_RUN。CPU小张量和fake-env不能保证GPU数值/调度/cache行为完全不变。
2. T/A_off可比阶段下实际成本：NOT_RUN。相同cohort不等于相同隐藏输入、GPU kernel重叠或同等资源争用；完整boot时钟、真实overlap及比例须从GPU原始账本核验。
3. 模型成功率非劣、环境步数改善或clean整局加速：未证明；本轮没有新真实episode或成本数字。
4. 原phase profile有CUDA同步，新增Python sidecar也有开销。server区间不含返回后sidecar构造/写入，其开销仍在client transport内；不把残差叫纯env。自然调度/profile-off/无屏障E2E是独立后续关卡。
5. 数据无新candidate-query teacher特征，不能把本轮部署动作或CPU合成fixture直接用于draft训练。KL/RL/大小模型联训仍延期。

## 5. 下一步（需另确认；本轮不执行）

**先确定一小轮真实GPU完整输出/轨迹不扰动验证的具体预算和开关基线。** 保持各自冻结T/A_off、同init/首局/种子/公共native+fingerprint；比较profile+包装前后完整action/video、已测state/RNG/RGB、命令与成功/步数。A_off旧评测的seed/首局历史不完全符合新合同，不能直接把旧轨迹当已验证相同基线。验证失败就停在首分叉，不追加、不把成本差归因训练。

关卡通过后，才**另批准**上述最多8完整T/A_off同负载诊断：逐请求成本/CFG/实际KV及物理阶段、匹配覆盖、single-demand尾部；之后才决定冻结target困难状态同query训练或其他预算优化。无屏障/profile-off自然E2E仍单独验证；L2辅助、任务成功/步数/长尾/work/E2E主验收，不自动RL/联训。

记录归属：本报告及RESEARCH第22节、Knowledge/四个索引/compact/PROGRESS/log。历史成本草案、第21节、旧raw/runtime/protected与tile记录保留，不改写历史为新实证。
