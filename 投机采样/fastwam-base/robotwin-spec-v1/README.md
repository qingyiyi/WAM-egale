# RoboTwin 投机接口准备

当前准备源码身份见 `SOURCE-PREPARED-v4.json`；此前SOURCE和CPU检查保留，不是已部署模型版本。

本目录是 CPU 已检查的迁移准备，未部署 GPU，未训练预测器，也没有 RoboTwin 成功率。canonical 作业仍使用独立 `robotwin-code-v1`，不会因本目录变化改变。

- `draft.py`：原轻量 Transformer 同一层、64 隐藏维、4 heads、256 FFN；按 32×14 动作改成 31 维 token 输入、28 维双方向输出，共 55,900 参数。899 个 feature 与 448 个 residual 归一化项必须由 RoboTwin FIT 重算。拒绝 LIBERO checkpoint schema。
- `adapter.py`：保持原 infer_action 的观测、文本、proprio、视觉 KV 和 scheduler，只替换动作循环。模型 `infer_action` 未被默认替换；部署需显式绑定。
- `verifier.py`：从已采用的 B1/B7 adapter 提取，只将动作通道改为 14；没有历史接受深度强制路径。7 个 batch 行保持独立 32-token 序列。
- `real_speculative.py`：K2/B2 的 1+2+4 状态、FP32 `abs(delta)*L2` 分数、原路径选择、bootstrap/fallback/bonus/尾部计费；每次完整调用仍提交 10 个 scheduler 更新。
- `train.py`：准备同一 final3000、FIT-only normalizer、CAL Q80 配方；要求已通过的 canonical 记录和独立新 panel。FIT/CAL/DEV/TEST 按完整 `group` 隔离，RoboTwin teacher schema 独立。当前没有 panel，不能直接训练。

`check_cpu_contract.py` 使用固定源码的真实 BF16 十步 shift5 scheduler 和可控合成 target，核对全拒绝、接受一层、接受两层、非零分支索引、尾部、bonus 与计数；三种路径都与各自合法 canonical 更新逐位一致，G=10。真实 target 的 B7/B1 数值和候选隔离仍须在加载模型后检查；CPU 结果不构成其替代。

后续依赖：四完整 canonical → 图像/动作/终态审阅 → 新 RoboTwin episode 分组与有限预算 → 采集真实 target 瞬时方向 → 固定训练/CAL → 三臂闭环。禁止使用 LIBERO tau，禁止把合成检查叫接受率或速度结果。官方 RoboTwin canonical 默认 eager，未来两臂需明确共用相同编译条件。

2026-10-27 04:33：`policy_bridge.py`从固定官方Robo wrapper提取调用前原语句，真实stats的CPU对比得到全部kwargs与动作后处理逐位一致，原观测未改。`server.py`已准备采集/TF/PF统一入口，启动须匹配已通过canonical审阅的SHA及新panel/预算；首真实观测仍要求官方wrapper、collector和所选compile路径核验。尚未部署或运行GPU。源码清单见SOURCE-BRIDGE-v2.json。

2026-10-27T04:40:30.592894+08:00：`client.py`已准备完整episode入口，沿固定expert初态筛选、官方horizon及每chunk24条命令。T记录有效scene seed和expert前Python RNG；S/P显式回放并检查生成指令一致，无效配对不能换seed冒充。每进程单episode、可变状态隔离，两客户端共享服务。`server.py`保存真实瞬时方向标签、首输入完整官方对照与B7数值/候选隔离/验收路径敏感性；`train.py`校验FIT/CAL/DEV/TEST两两组隔离。新CPU边界检查v2通过，尚未开始任何Robo训练或闭环。

部署前数据约束：Robo专家有效性筛选可能把多个请求起始seed推进到同一实际有效seed。采集controller必须按上一实际有效seed继续、并去重task/config/actual_scene_seed；FIT/CAL/DEV的候选seed区间或实际初态组也需互斥，clean/randomized相关scene不能跨split。v4训练入口额外检查实际FIT/CAL场景，不仅比较group名字；DEV实际场景去重需在闭环汇总核验，不能靠名称认为已隔离。

后续Robo supervisor注意：大量episode应按两个固定worker槽复用各自编译缓存，环境/RNG/输出仍逐episode隔离；不要为上万episode复制CUDA/Warp编译缓存。当前四canonical的逐task缓存是有界pilot特例。`review_canonical.py`需作为实际文件复制后执行（以`__file__`记录源码SHA），按预登记协议核对完整动作前缀和图片联系表；它只生成技术审计，最终visual review由当前agent完成，不启动训练。
